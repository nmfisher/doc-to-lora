"""Offline LoRA-tensor-diff probe against a trained checkpoint.

The in-training callback (src/ctx_to_lora/callbacks.py:CtxSensitivityProbe)
only fires on save_step, and a running training process can't pick up code
changes — so when we extend the probe we lose the ability to interrogate
already-running runs. This script runs the same probe logic against a
checkpoint directory on disk, independent of any live training. Use it to:

  - decide whether to keep training a run that's showing 1/3 distinct
    generation outputs (is the hypernet producing distinct LoRAs that the
    decoder is collapsing, or are the LoRAs themselves identical?)
  - re-analyse historical checkpoints after a probe upgrade
  - compare two checkpoints (e.g., step 500 vs step 5000) to see whether
    the diff/norm ratio is moving in the right direction over training

What gets reported per (module, A|B):
  - Frobenius norm per context
  - pairwise L2 distance between contexts
  - diff/norm ratio (~0 = collapsed / context-blind, ~0.5+ = distinct)

Usage on the pod (or any GPU host):
  uv run python scripts/probe_checkpoint.py \\
      --checkpoint train_outputs/runs/<run>_<hash>/checkpoint-500 \\
      --target-modules down_proj --ctx-encoder-last-layer 20

Local (Mac, CPU): same command + `--device cpu`. Loads Gemma 4 E2B + a
truncated ctx_encoder + the trained hypernet (~10-12 GB of weights at
fp32). ~24 GB unified-memory Mac handles it; smaller will swap.
"""

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig

from ctx_to_lora.configs import (
    AggregatorArguments,
    CtxEncoderArguments,
    HypernetArguments,
)
from ctx_to_lora.modeling.ctx_encoder import CTX_ENCODER_TYPE
from ctx_to_lora.model_loading import (
    get_lora_config,
    get_model_and_tokenizer,
    get_tokenizer,
)
from ctx_to_lora.modeling.hypernet import (
    ModulatedPretrainedModel,
    get_hypernet_config,
)


# Same probe definitions as the in-training callback.
PROMPT = (
    "What is the time complexity of the function shown in the context, "
    "and how much auxiliary memory does it use?"
)

CONTEXTS = {
    "fib_iter": """\
def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number iteratively.

    Uses constant memory (a pair of running totals) and O(n) time.
    \"\"\"
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",
    "fib_recur": """\
def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number recursively.

    No memoization, so identical subproblems are recomputed down to
    the base case. F(n) = F(n-1) + F(n-2).
    \"\"\"
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
    "quicksort": """\
def quicksort(arr):
    \"\"\"Sort using Lomuto partition. Average O(n log n), worst O(n^2).\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[-1]
    less = [x for x in arr[:-1] if x <= pivot]
    greater = [x for x in arr[:-1] if x > pivot]
    return quicksort(less) + [pivot] + quicksort(greater)
""",
}


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the LoRA-tensor-diff probe against a saved checkpoint."
    )
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Path to a checkpoint-N directory (must contain "
                    "model.safetensors).")
    ap.add_argument("--model", default="google/gemma-4-E2B-it",
                    help="Base + ctx-encoder model id. Must match the model "
                    "the checkpoint was trained against.")
    ap.add_argument("--target-modules", default="down_proj",
                    help="Comma-separated list — must match the training "
                    "config (otherwise the hypernet shape doesn't match "
                    "the checkpoint state_dict).")
    ap.add_argument("--ctx-encoder-last-layer", type=int, default=20,
                    help="Same constraint as --target-modules: must match "
                    "training. 20 for down_proj only, 16 for all 5 modules.")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--layer-idx", type=int, default=8,
                    help="ctx_encoder layer to read activations from.")
    ap.add_argument("--n-latent-queries", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=9)
    ap.add_argument("--device", default=None,
                    help="cuda | cpu. Default: cuda if available.")
    args = ap.parse_args()

    if not args.checkpoint.is_dir():
        raise SystemExit(f"--checkpoint not a directory: {args.checkpoint}")
    safetensors_path = args.checkpoint / "model.safetensors"
    if not safetensors_path.is_file():
        raise SystemExit(f"missing {safetensors_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_modules = [m.strip() for m in args.target_modules.split(",")
                      if m.strip()]
    print(f"[probe] checkpoint={args.checkpoint}")
    print(f"[probe] target_modules={target_modules}, "
          f"ctx_encoder_last_layer={args.ctx_encoder_last_layer}, "
          f"device={device}")

    # Rebuild the model architecturally identical to overfit_single.py /
    # train.py so the checkpoint's hypernet weights can be loaded in.
    section("Step 1: build modulated model architecture")
    peft_config = get_lora_config(
        args.model, lora_r=args.lora_r, lora_dropout=0.0,
        target_modules=target_modules, lora_alpha=args.lora_alpha,
    )
    base_model, _ = get_model_and_tokenizer(
        model_name_or_path=args.model,
        train=True,
        requires_grad=False,
        use_flash_attn=False,
        peft_config=peft_config,
    )
    ctx_tokenizer = get_tokenizer(args.model)

    ctx_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(ctx_cfg, "text_config"):
        ctx_cfg = ctx_cfg.text_config

    ctx_encoder_args = CtxEncoderArguments(
        ctx_encoder_model_name_or_path=args.model,
        ctx_encoder_type=CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS,
        layer_idx=args.layer_idx,
        ctx_encoder_last_layer=args.ctx_encoder_last_layer,
    )
    hypernet_args = HypernetArguments(
        per_rank_gen=True, per_layer_processing=True,
    )
    aggregator_args = AggregatorArguments(
        n_latent_queries=args.n_latent_queries,
        num_blocks=args.num_blocks,
        num_self_attn_per_block=0,
    )
    hypernet_config = get_hypernet_config(
        base_model, ctx_cfg, hypernet_args, aggregator_args,
        ctx_encoder_args, peft_config=peft_config,
    )
    model = ModulatedPretrainedModel(
        base_model, hypernet_config, ctx_encoder_args,
        use_sequence_packing=False,
    )

    # The HF Trainer saves only trainable params via the safetensors writer,
    # so missing keys covers all frozen base/ctx params — expected.
    section("Step 2: load trained hypernet weights from checkpoint")
    state = load_file(str(safetensors_path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[probe] loaded {len(state)} tensors; "
          f"missing={len(missing)} (frozen base/ctx — expected), "
          f"unexpected={len(unexpected)}")
    if unexpected:
        # Anything unexpected is a config mismatch — fail loud rather than
        # quietly run with mis-shaped tensors.
        for k in unexpected[:5]:
            print(f"  unexpected key: {k}")
        raise SystemExit("checkpoint has tensors the model doesn't expect — "
                         "target-modules / ctx-encoder-last-layer / lora-r "
                         "likely don't match the training config.")

    model = model.to(device)
    model.eval()

    # ctx_encoder.base_model.name_or_path is empty after the wrap; internalize()
    # needs it to load the tokenizer (matches overfit_single.py:354 + the
    # callback's pre-probe fix).
    if not getattr(model.ctx_encoder.base_model, "name_or_path", ""):
        model.ctx_encoder.base_model.name_or_path = args.model

    section("Step 3: internalize each context, capture generated LoRAs")
    loras: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for label, ctx_text in CONTEXTS.items():
        model.reset()
        with torch.no_grad():
            model.internalize(ctx_text)
        gen = model.generated_loras
        loras[label] = {
            mname: {
                "A": gen[mname]["A"].detach().float().cpu().clone(),
                "B": gen[mname]["B"].detach().float().cpu().clone(),
            }
            for mname in gen
        }
        # Quick sanity: tensor shapes (should be [1, n_layers, r, d_in]
        # for A and [1, n_layers, d_out, r] for B, or similar).
        first_mod = next(iter(gen))
        print(f"[probe] {label}: A.shape={tuple(gen[first_mod]['A'].shape)}, "
              f"B.shape={tuple(gen[first_mod]['B'].shape)}")
    model.reset()

    section("Step 4: per-module Frobenius norms + pairwise diffs")
    labels = list(loras.keys())
    pairs = [
        (labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    # Aggregate stats across all modules to print a single bottom-line ratio.
    total_avg_norm = 0.0
    total_avg_diff = 0.0
    n_module_tensors = 0
    for mname in loras[labels[0]]:
        print(f"\nModule: {mname}")
        for tname in ("A", "B"):
            norms = {
                label: loras[label][mname][tname].flatten().norm().item()
                for label in labels
            }
            diffs = {
                f"{a}_vs_{b}": (
                    loras[a][mname][tname] - loras[b][mname][tname]
                ).flatten().norm().item()
                for a, b in pairs
            }
            avg_norm = sum(norms.values()) / len(norms)
            avg_diff = sum(diffs.values()) / len(diffs)
            ratio = avg_diff / avg_norm if avg_norm > 0 else 0.0
            norms_str = " ".join(f"{l}={v:.3f}" for l, v in norms.items())
            diffs_str = " ".join(f"{k}={v:.3f}" for k, v in diffs.items())
            print(f"  {tname}: norms[{norms_str}] | "
                  f"pairwise[{diffs_str}] | diff/norm={ratio:.4f}")
            total_avg_norm += avg_norm
            total_avg_diff += avg_diff
            n_module_tensors += 1

    section("Verdict")
    overall = (total_avg_diff / total_avg_norm) if total_avg_norm > 0 else 0.0
    print(f"Overall diff/norm ratio (mean across all (module, A|B)): "
          f"{overall:.4f}")
    print(
        "  ratio < 0.01     → near-perfect LoRA collapse; the hypernet is "
        "  emitting essentially the same tensor regardless of context.\n"
        "  ratio 0.01-0.10  → small differences; greedy decoding will almost "
        "  certainly produce identical outputs.\n"
        "  ratio 0.10-0.50  → meaningful differences; if greedy still "
        "  collapses, the failure is downstream of the LoRA (weak signal "
        "  vs. the base model's prior on the prompt).\n"
        "  ratio > 0.50     → strongly distinct LoRAs; greedy should be "
        "  picking different tokens."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
