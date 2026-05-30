import ast
import gc
import hashlib
import logging
import os
import random
import string
import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from enum import Enum
from operator import attrgetter

import torch
import yaml
from peft import PeftConfig, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils import get_peft_model_state_dict

TRAINING_TASK = Enum("TRAINING_TASK", ["CAUSAL_LM", "COMPLETION"])


logger = logging.getLogger()


# taken from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
@contextmanager
def evaluating(*models):
    """Temporarily switch to evaluation mode."""
    is_training = [model.training if model is not None else False for model in models]
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield models
    finally:
        for model, training in zip(models, is_training):
            if model is not None:
                model.train(training)


def get_layers(model):
    if hasattr(model, "model"):
        return get_layers(model.model)
    return model.layers


def get_num_layers(model):
    return len(get_layers(model))


_ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj")
_MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")


def get_layer_dim_profile(layer, target_modules) -> tuple:
    """Hashable signature of (in_features, out_features) per target_module in this layer.
    Missing modules are encoded as (mname, None, None) so they don't collapse with
    layers that do have the module."""
    if not isinstance(target_modules, (list, tuple, set)):
        return ()
    profile = []
    for mname in sorted(target_modules):
        if mname in _ATTN_TARGETS:
            path = f"self_attn.{mname}"
        elif mname in _MLP_TARGETS:
            path = f"mlp.{mname}"
        else:
            profile.append((mname, None, None))
            continue
        try:
            module = attrgetter(path)(layer)
        except AttributeError:
            module = None
        if isinstance(module, torch.nn.Linear):
            profile.append((mname, module.in_features, module.out_features))
        else:
            profile.append((mname, None, None))
    return tuple(profile)


def select_majority_dim_group(model, target_modules) -> tuple[int, ...]:
    """Group layers by dim profile across target_modules; return indices of the
    largest group as a tuple of ints. Tie-break by lowest first-layer index.

    For homogeneous models (Gemma 2, Llama, etc.) every layer has the same
    profile, so this returns tuple(range(num_layers)) — identical to the old
    arange(num_layers) behavior. For heterogeneous models (Gemma 4 has 4
    dim profiles across 35 layers) this returns the largest profile's indices,
    typically non-contiguous."""
    layers = get_layers(model)
    n_layers = len(layers)
    if not isinstance(target_modules, (list, tuple, set)):
        return tuple(range(n_layers))
    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, layer in enumerate(layers):
        groups[get_layer_dim_profile(layer, target_modules)].append(idx)
    best_profile = max(
        groups.keys(),
        key=lambda p: (len(groups[p]), -groups[p][0]),
    )
    indices = tuple(groups[best_profile])
    if len(groups) > 1:
        logger.info(
            f"Layer-dim grouping: {len(groups)} distinct profiles across "
            f"{n_layers} layers; selected majority group of {len(indices)} layer(s)."
        )
        for p, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            marker = " <-- selected" if p == best_profile else ""
            logger.debug(f"  group ({len(idxs)} layers) layers={idxs}{marker}")
    # Sanity: any None dims in the selected profile signal a target_module that
    # doesn't exist on these layers (would crash attrgetter at apply time).
    missing = [m for (m, d_in, d_out) in best_profile if d_in is None or d_out is None]
    if missing:
        logger.warning(
            f"target_modules {missing} not found on selected layers — drop them "
            f"from target_modules in your config."
        )
    return indices


def get_base_model(model):
    if hasattr(model, "model"):
        return get_base_model(model.model)
    return model


def get_num_params(model):
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    return total_params, trainable_params


def log_num_train_params(model):
    logger.debug("Trainable model parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.debug(f"{name}, dtype:{p.dtype}")

    num_total_params, num_trainable_params = get_num_params(model)
    logger.info(
        f"trainable params: {num_trainable_params:,d} "
        f"|| all params: {num_total_params:,d} "
        f"|| trainable%: {100 * num_trainable_params / num_total_params:.4f}"
    )


def get_run_name(seed_str: str | None = None):
    if not seed_str:
        uuid = "".join(
            [random.choice(string.ascii_letters + string.digits) for _ in range(8)]
        )
        run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"
    else:
        # Generate a UUID from the seed string
        hash_object = hashlib.sha256(seed_str.encode())
        uuid = hash_object.hexdigest()[:8]  # Take the first 8 characters of the hash
        run_name = seed_str + f"_{uuid}"
    return run_name


def try_convert(s):
    try:
        return ast.literal_eval(s)
    except:
        return s


def extract_cli_args(argv: list[str]):
    out = dict()
    for elem in argv:
        if elem.endswith(".yaml"):
            out["config"] = elem

        elif elem.startswith("--"):
            k, v = elem.split("=")
            k = k.split("--")[1]
            v = try_convert(v)
            # if k.startswith('env_'):
            #     k = k.split('_')[1]
            out[k] = v
    return out


def setup_logging(output_dir, debug=False):
    global logger

    os.makedirs(output_dir, exist_ok=True)

    log_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_level = logging.DEBUG if debug else logging.INFO
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)

    log_path = f"{output_dir}/debug.log"
    debug_handler = logging.FileHandler(log_path, delay=True)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(log_formatter)
    logger.addHandler(debug_handler)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Logging to: {log_path}")


def validate_args(args_list):
    # there shouldn't be overlap between args
    keys = set()
    for args in args_list:
        logger.debug(args)
        args_keys = set(vars(args).keys())
        assert len(keys & args_keys) == 0, "Overlap between args"
        keys |= args_keys


def save_yaml(data, path):
    # Filter out non-primitive fields
    data = {
        k: v
        for k, v in data.items()
        if isinstance(v, (int, float, str, bool, list, dict, type(None)))
    }

    with open(path, "w") as file:
        yaml.dump(data, file)


def get_peft_modules(model, peft_config: PeftConfig) -> list[dict[str, str]]:
    # Non-PEFT path (Gemma 4): the model is bare, so target modules are plain
    # nn.Linear instances rather than BaseTunerLayer wrappers.
    leaf_type = BaseTunerLayer if isinstance(model, PeftModel) else torch.nn.Linear
    return [
        {"name": name, "module": module}
        for name, module in model.named_modules()
        if name.split(".")[-1] in peft_config.target_modules
        and isinstance(module, leaf_type)
        and check_target_module_exists(peft_config, name)
    ]


def get_peft_in_out_features(
    model,
    peft_config: PeftConfig | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    if peft_config is None:
        return None, None
    is_peft = isinstance(model, PeftModel)
    in_features = dict()
    out_features = dict()
    for module_info in get_peft_modules(model, peft_config):
        module_name = module_info["name"]
        module = module_info["module"]
        # In PEFT mode the LoRA wrapper exposes `.base_layer`; in the bare-model
        # path (Gemma 4) the module IS the Linear leaf.
        leaf = module.base_layer if is_peft else module
        assert isinstance(leaf, torch.nn.Linear), (
            "all modules should be a leaf Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules

        if name not in in_features:
            in_features[name] = module.in_features
            out_features[name] = module.out_features
        else:
            # assumes each module has the same input and output features
            assert in_features[name] == module.in_features
            assert out_features[name] == module.out_features

    return in_features, out_features


def generated_lora_to_state_dict(
    lora_dict: dict,
    module_names: dict,
    target_modules: list[str],
    layer_indices: Iterable[int],
) -> dict:
    # Position-based indexing: module_names[t] and lora_dict[t]["A"/"B"] are both
    # sized len(layer_indices). For contiguous layer_indices == range(N) this is
    # identical to the old raw-index scheme; required for non-contiguous (Gemma 4).
    layer_indices = list(layer_indices)
    lora_state_dict = dict()
    for target_module in target_modules:
        for pos, _layer_idx in enumerate(layer_indices):
            for module_name in module_names[target_module][pos]:
                if "lora_A" in module_name:
                    lora_state_dict[module_name] = (
                        lora_dict[target_module]["A"][pos].cpu().contiguous()
                    )
                elif "lora_B" in module_name:
                    lora_state_dict[module_name] = (
                        lora_dict[target_module]["B"][pos].cpu().contiguous()
                    )
                else:
                    raise ValueError(f"Unexpected module name: {module_name}")
    return lora_state_dict


def get_lora_module_names(
    model,
    target_modules: list[str],
    layer_indices: Iterable[int],
) -> dict[str, list[str]]:
    layer_indices = list(layer_indices)
    module_names = {
        target_module: [[] for _ in range(len(layer_indices))]
        for target_module in target_modules
    }
    if isinstance(model, PeftModel):
        for k in get_peft_model_state_dict(model):
            if "lora" not in k:
                continue
            layer_idx = int(k.split("layers.")[-1].split(".")[0])
            if layer_idx in layer_indices:
                pos = layer_indices.index(layer_idx)
                for target_module in target_modules:
                    if target_module in k:
                        module_names[target_module][pos].append(k)
                        break
    else:
        # Non-PEFT path (Gemma 4): the bare model has no LoRA submodules, so
        # synthesize the expected lora_A/lora_B key names from each target Linear.
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            leaf = name.split(".")[-1]
            if leaf not in target_modules:
                continue
            if "layers." not in name:
                continue
            try:
                layer_idx = int(name.split("layers.")[-1].split(".")[0])
            except ValueError:
                continue
            if layer_idx not in layer_indices:
                continue
            pos = layer_indices.index(layer_idx)
            module_names[leaf][pos].append(f"{name}.lora_A.weight")
            module_names[leaf][pos].append(f"{name}.lora_B.weight")
    return module_names


def compile_linear(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.compile()


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()


def concat_list(l):
    out = []
    for x in l:
        out += x
    return out


def check_is_iterable(x):
    try:
        iter(x)
    except TypeError:
        return False
    return True
