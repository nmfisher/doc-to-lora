"""Test a trained Gemma 4 D2L hypernet on a held-out context.

Loads the latest checkpoint under train_outputs/runs/<run_name>/, constructs
the modulated model fresh (matching training-time setup), loads the
hypernet weights, then runs the canonical "does this learn anything"
comparison:

    1. Baseline: ask the base Gemma 4 a question with no LoRA.
    2. Internalized: feed it a context via model.internalize(); the
       hypernet generates a LoRA from the context, applies it, and we
       ask the same question. The expectation is that the answer should
       be informed by the context.

Run inside the doc-to-lora container:
    python scripts/test_gemma4_hypernet.py --run-name gemma4_train_v1

Args:
    --run-name      train_outputs/runs/ subdir to load checkpoint from
    --checkpoint    explicit checkpoint path (overrides --run-name auto-pick)
    --context-file  path to a text file used as the "internalized" context
                    (default: a small synthetic Python snippet)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig

sys.path.insert(0, "src")

from ctx_to_lora.configs import (
    AggregatorArguments,
    CtxEncoderArguments,
    HypernetArguments,
)
from ctx_to_lora.model_loading import get_lora_config, get_model_and_tokenizer
from ctx_to_lora.modeling.ctx_encoder import CTX_ENCODER_TYPE
from ctx_to_lora.modeling.hypernet import (
    ModulatedPretrainedModel,
    get_hypernet_config,
)

MODEL = os.environ.get("MODEL_DIR", "google/gemma-4-E2B-it")
TARGET_MODULES = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_CONTEXT = """\
def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number iteratively.

    The series starts with F(0) = 0, F(1) = 1, then each successive value is
    the sum of the previous two. This implementation uses constant memory
    (just a pair of running totals) and O(n) time.
    \"\"\"
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""

DEFAULT_QUESTION = (
    "What is the time complexity of the fibonacci function shown in the "
    "context, and how much auxiliary memory does it use?"
)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def find_latest_checkpoint(run_dir: Path) -> Path:
    """Find the highest-step `checkpoint-N` subdir under run_dir."""
    candidates = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not candidates:
        sys.exit(f"No checkpoint-* dirs found under {run_dir}")
    return candidates[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--context-file", type=str, default=None)
    p.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument(
        "--scaler-b-boost",
        type=float,
        default=1.0,
        help=(
            "Inference-time multiplier applied to all scaler_B params before "
            "internalize(). 1.0 = unchanged; 10 ≈ simulates lora_alpha bumped 10×; "
            "100 ≈ what the hypernet 'would' do after a lot more training."
        ),
    )
    args = p.parse_args()

    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
    elif args.run_name:
        ckpt_dir = find_latest_checkpoint(Path("train_outputs/runs") / args.run_name)
    else:
        sys.exit("Pass either --run-name or --checkpoint")

    print(f"[test] loading from {ckpt_dir}")

    # Find the weight file the trainer wrote. Newer transformers saves to
    # model.safetensors by default; older runs may have pytorch_model.bin.
    weights_path = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (ckpt_dir / name).exists():
            weights_path = ckpt_dir / name
            break
    if weights_path is None:
        sys.exit(f"No model weights found in {ckpt_dir}")
    print(f"[test] weights: {weights_path}")

    section("Step 1: construct fresh modulated model (matches training setup)")
    # Read the saved training args so AggregatorArguments / HypernetArguments
    # match exactly. Without this we hit shape mismatches like latents_q
    # (8 in YAML vs 208 in dataclass default).
    args_yaml_path = ckpt_dir.parent / "args.yaml"
    if not args_yaml_path.exists():
        # Older runs may store at run root rather than per-checkpoint
        args_yaml_path = ckpt_dir.parent / "args.yaml"
    saved_args = {}
    if args_yaml_path.exists():
        with args_yaml_path.open() as f:
            # args.yaml contains Python object tags like
            # !!python/object/apply:transformers.trainer_utils.IntervalStrategy
            # so safe_load chokes. We trust this file (we wrote it), so use
            # UnsafeLoader to deserialize Python objects.
            saved_args = yaml.load(f, Loader=yaml.UnsafeLoader) or {}
        print(f"[test] loaded training args from {args_yaml_path}")
    else:
        print(f"[test] WARNING: {args_yaml_path} not found; using dataclass defaults")

    def pick(key, default):
        return saved_args.get(key, default)

    peft_config = get_lora_config(
        MODEL,
        lora_r=pick("lora_r", 8),
        lora_dropout=pick("lora_dropout", 0.0),
        target_modules=pick("target_modules", TARGET_MODULES),
        lora_alpha=pick("lora_alpha", 8),
    )
    base_model, tokenizer = get_model_and_tokenizer(
        model_name_or_path=MODEL,
        train=False,
        requires_grad=False,
        use_flash_attn=False,
        peft_config=peft_config,
    )

    ctx_cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if hasattr(ctx_cfg, "text_config"):
        ctx_cfg = ctx_cfg.text_config

    ctx_encoder_args = CtxEncoderArguments(
        ctx_encoder_model_name_or_path=MODEL,
        ctx_encoder_type=pick("ctx_encoder_type", CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS),
        layer_idx=pick("layer_idx", 8),
        ctx_encoder_last_layer=pick("ctx_encoder_last_layer", None),
    )
    hypernet_args = HypernetArguments(
        per_rank_gen=pick("per_rank_gen", True),
        per_layer_processing=pick("per_layer_processing", True),
    )
    aggregator_args = AggregatorArguments(
        n_latent_queries=pick("n_latent_queries", 8),
        num_blocks=pick("num_blocks", 9),
        num_self_attn_per_block=pick("num_self_attn_per_block", 0),
    )
    hypernet_config = get_hypernet_config(
        base_model,
        ctx_cfg,
        hypernet_args,
        aggregator_args,
        ctx_encoder_args,
        peft_config=peft_config,
    )

    model = ModulatedPretrainedModel(
        base_model,
        hypernet_config,
        ctx_encoder_args,
        use_sequence_packing=False,  # for generation
    )
    model.eval()

    section("Step 2: load trained hypernet weights")
    # Load the saved tensors. The Trainer's default save_strategy=epoch
    # writes the modulated model's state_dict — which, with our metadata
    # gating, is hypernet-tensors-only.
    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cuda")
    else:
        state_dict = torch.load(str(weights_path), map_location="cuda", weights_only=False)

    # Filter to hypernet keys only, in case the file holds extras (e.g.
    # base-model name strings from an older save format).
    tensor_only = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
    missing, unexpected = model.hypernet.load_state_dict(tensor_only, strict=False)
    print(f"[test] loaded {len(tensor_only)} tensors")
    print(f"[test] missing keys: {len(missing)}  unexpected: {len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:3]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:3]}")

    if args.scaler_b_boost != 1.0:
        n_boosted = 0
        with torch.no_grad():
            for name, p in model.hypernet.named_parameters():
                if "scaler_B" in name:
                    p.mul_(args.scaler_b_boost)
                    n_boosted += 1
        print(
            f"[test] BOOSTED {n_boosted} scaler_B params by {args.scaler_b_boost}× "
            f"(simulates higher effective lora_alpha at inference time)"
        )

    section("Step 3: load test context + question")
    if args.context_file:
        context_str = Path(args.context_file).read_text()
    else:
        context_str = DEFAULT_CONTEXT
    question_str = args.question
    print(f"[test] context ({len(context_str)} chars):")
    print(context_str.rstrip()[:500] + (" ..." if len(context_str) > 500 else ""))
    print(f"\n[test] question: {question_str}")

    # Tokenize the question via the chat template.
    chat_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": question_str}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to("cuda")

    section("Step 4: baseline — base Gemma 4, no LoRA")
    # ModulatedPretrainedModel._init_model patches every target Linear's
    # forward to lora_forward, expecting A/B kwargs that get bound later
    # by apply_lora_to_layers. For the baseline call we want the
    # un-patched original forwards, so reset() first (restores
    # forward = forward_orig per module).
    model.reset()
    with torch.no_grad():
        baseline_out = model.base_model.generate(
            input_ids=chat_inputs["input_ids"],
            attention_mask=chat_inputs.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    baseline_text = tokenizer.decode(
        baseline_out[0, chat_inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    print(baseline_text.strip())

    section("Step 5: with internalized context (hypernet-generated LoRA)")
    # model.internalize() calls get_tokenizer(self.ctx_encoder.base_model.name_or_path),
    # but PerLayerActivations stashes the inner Gemma4TextModel whose
    # name_or_path got cleared. Restore it from MODEL so the tokenizer
    # lookup succeeds.
    if not getattr(model.ctx_encoder.base_model, "name_or_path", ""):
        model.ctx_encoder.base_model.name_or_path = MODEL
    model.internalize(context_str)

    with torch.no_grad():
        modulated_out = model.generate(
            input_ids=chat_inputs["input_ids"],
            attention_mask=chat_inputs.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    modulated_text = tokenizer.decode(
        modulated_out[0, chat_inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    print(modulated_text.strip())

    model.reset()

    section("Summary")
    print("Baseline answer length :", len(baseline_text.strip()))
    print("Modulated answer length:", len(modulated_text.strip()))
    print(
        "Identical?              :",
        "yes" if baseline_text.strip() == modulated_text.strip() else "no",
    )


if __name__ == "__main__":
    main()
