"""Diagnose why gen_lora_l1_norm stays pinned at ~0.0012 during training.

Runs four checks against a trained checkpoint:

  [1] Scaler stats: load the hypernet state_dict, print the values of all
      learnable amplitude scalers (scaler_A, scaler_B, etc.) and any other
      params whose name suggests "amplitude" or "norm". If these cluster
      near 1e-3, that explains a pinned LoRA L1 norm — the scalers ARE
      the cap.

  [2] Wiring check: forward-pass the base model with (a) no LoRA via
      model.reset() and (b) a hypernet-generated LoRA via
      model.internalize(ctx). Compare the resulting logits. If they're
      identical (or differ by < 1e-5 in mean abs), the LoRA isn't
      reaching the base model's linear forwards — wiring bug.

  [3] LoRA magnitude per-layer: print the L1/L2/max-abs of each generated
      LoRA A and B tensor for one context. Tells us whether the pinning
      is uniform across layers or specific layers are at zero.

  [4] Base-model-only CE on a training sample: if the cocoon QA can be
      answered by the base model alone (i.e. CE is already low), the
      hypernet has no gradient pressure to grow useful LoRA. This
      separates "training is broken" from "the task doesn't need a LoRA".

Run inside the container, same volumes as test_hypernet:
    python scripts/debug_hypernet.py --checkpoint <path-to-checkpoint-dir>
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

DEBUG_CONTEXT = """\
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""

DEBUG_QUESTION = "What does this function compute?"


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_saved_args(args_yaml_path: Path) -> dict:
    if not args_yaml_path.exists():
        print(f"[debug] WARNING: {args_yaml_path} not found; using dataclass defaults")
        return {}
    with args_yaml_path.open() as f:
        # args.yaml has Python object tags (IntervalStrategy etc.) — UnsafeLoader handles them.
        return yaml.load(f, Loader=yaml.UnsafeLoader) or {}


def construct_model(saved_args: dict):
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
        use_sequence_packing=False,
    )
    model.eval()
    return model, tokenizer


def load_weights(model: ModulatedPretrainedModel, weights_path: Path) -> None:
    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(weights_path), device="cuda")
    else:
        sd = torch.load(str(weights_path), map_location="cuda", weights_only=False)
    tensors_only = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    missing, unexpected = model.hypernet.load_state_dict(tensors_only, strict=False)
    print(f"[debug] loaded {len(tensors_only)} tensors  missing={len(missing)}  unexpected={len(unexpected)}")


def check1_scalers(model: ModulatedPretrainedModel) -> None:
    section("[1] Scaler / amplitude param stats")
    # Anything that suggests an amplitude knob: name contains scaler,
    # scale, alpha, gain. Print stats so we can spot pinned-near-zero values.
    SUSPECTS = ("scaler", "scale", "alpha", "gain")
    rows = []
    for name, p in model.hypernet.named_parameters():
        if any(s in name.lower() for s in SUSPECTS):
            rows.append((
                name,
                tuple(p.shape),
                float(p.detach().abs().mean()),
                float(p.detach().abs().max()),
                float(p.detach().abs().min()),
            ))
    if not rows:
        print("(no params matched scaler/scale/alpha/gain)")
    else:
        print(f"{'name':<70}  {'shape':<14}  {'|mean|':>10}  {'|max|':>10}  {'|min|':>10}")
        print("-" * 122)
        for name, shape, am, mx, mn in rows:
            print(f"{name:<70}  {str(shape):<14}  {am:>10.4e}  {mx:>10.4e}  {mn:>10.4e}")
    # Verdict: if |mean| of scalers is ~1e-3, that's the cap.
    if rows:
        avg = sum(r[2] for r in rows) / len(rows)
        print(f"\n[verdict] mean |scaler| across {len(rows)} params: {avg:.4e}")
        if avg < 1e-2:
            print("          → scalers are pinned small; this likely caps gen_lora_l1_norm.")
        else:
            print("          → scalers are not the bottleneck.")


def check2_wiring(model: ModulatedPretrainedModel, tokenizer) -> None:
    section("[2] LoRA wiring: does internalize() actually change logits?")
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": DEBUG_QUESTION}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to("cuda")

    def first_token_logits(use_lora: bool) -> torch.Tensor:
        # We can't call base_model(...) directly:
        # ModulatedPretrainedModel._init_model patches every target Linear
        # to lora_forward, which expects A/B kwargs bound at generate-time
        # via apply_lora_to_layers. Going through the modulated generate
        # is the supported path that binds (or skips) A/B correctly.
        # Generate exactly 1 token and grab its scores.
        if use_lora:
            if not getattr(model.ctx_encoder.base_model, "name_or_path", ""):
                model.ctx_encoder.base_model.name_or_path = MODEL
            model.internalize(DEBUG_CONTEXT)
            out = model.generate(
                input_ids=chat["input_ids"],
                attention_mask=chat.get("attention_mask"),
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:
            model.reset()
            # base_model.generate with no patched forwards = plain Gemma 4
            out = model.base_model.generate(
                input_ids=chat["input_ids"],
                attention_mask=chat.get("attention_mask"),
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        # scores is a tuple of length max_new_tokens, each (1, vocab)
        return out.scores[0].clone()

    with torch.no_grad():
        logits_base = first_token_logits(use_lora=False)
        logits_lora = first_token_logits(use_lora=True)
    model.reset()

    abs_diff = (logits_lora - logits_base).abs()
    rel_diff = abs_diff / (logits_base.abs() + 1e-8)
    print(f"logits shape: {tuple(logits_base.shape)}")
    print(f"base logits mean abs : {logits_base.abs().mean().item():.6f}")
    print(f"lora logits mean abs : {logits_lora.abs().mean().item():.6f}")
    print(f"abs diff mean        : {abs_diff.mean().item():.6e}")
    print(f"abs diff max         : {abs_diff.max().item():.6e}")
    print(f"rel diff mean        : {rel_diff.mean().item():.6e}")
    identical = torch.allclose(logits_base, logits_lora)
    print(f"torch.allclose       : {identical}")
    if abs_diff.mean().item() < 1e-5:
        print("[verdict] LoRA effect on logits is negligible (<1e-5 mean abs).")
        print("          Either wiring is broken OR generated LoRA is effectively zero.")
    else:
        print("[verdict] LoRA changes logits — wiring is functional.")


def check3_generated_lora_magnitudes(model: ModulatedPretrainedModel) -> None:
    section("[3] Generated LoRA per-layer magnitudes")
    if not getattr(model.ctx_encoder.base_model, "name_or_path", ""):
        model.ctx_encoder.base_model.name_or_path = MODEL
    model.internalize(DEBUG_CONTEXT)
    loras = model.generated_loras
    if loras is None:
        print("model.generated_loras is None — internalize() did not store loras.")
        model.reset()
        return

    # generated_loras is a list/dict of (A, B) per layer-and-module.
    # Print summary per layer.
    def stats(t):
        t = t.detach()
        return (t.abs().mean().item(), t.abs().max().item(), t.shape)

    print(f"type(generated_loras): {type(loras).__name__}")
    if isinstance(loras, dict):
        for k, v in list(loras.items())[:10]:
            if hasattr(v, "shape"):
                m, mx, sh = stats(v)
                print(f"  {k}: shape={sh}  |mean|={m:.4e}  |max|={mx:.4e}")
            else:
                print(f"  {k}: type={type(v).__name__}  (not a tensor)")
    elif isinstance(loras, (list, tuple)):
        for i, item in enumerate(loras[:5]):
            print(f"  loras[{i}]: type={type(item).__name__}")
            if isinstance(item, (list, tuple)):
                for j, sub in enumerate(item):
                    if hasattr(sub, "shape"):
                        m, mx, sh = stats(sub)
                        print(f"    [{j}]: shape={sh}  |mean|={m:.4e}  |max|={mx:.4e}")
            elif hasattr(item, "shape"):
                m, mx, sh = stats(item)
                print(f"    shape={sh}  |mean|={m:.4e}  |max|={mx:.4e}")
            elif isinstance(item, dict):
                for k, v in list(item.items())[:3]:
                    if hasattr(v, "shape"):
                        m, mx, sh = stats(v)
                        print(f"    {k}: shape={sh}  |mean|={m:.4e}  |max|={mx:.4e}")
    else:
        print(f"  unexpected type: {type(loras)}")
    model.reset()


def check4_base_ce_on_training_sample(model: ModulatedPretrainedModel, tokenizer) -> None:
    section("[4] CE loss of base model on a training sample (no LoRA)")
    # Pull the first row from the prepared training parquet (after packing
    # was run earlier — we re-tokenize a raw QA pair from queries.jsonl
    # to avoid depending on prepared dataset state).
    queries_path = Path("data/raw_datasets/cocoon_code_qa/queries.jsonl")
    if not queries_path.exists():
        # Fall back to whatever raw queries we can find on the volume.
        candidates = list(Path("data/raw_datasets").rglob("queries.jsonl"))
        if candidates:
            queries_path = candidates[0]
    if not queries_path.exists():
        print("queries.jsonl not found anywhere under data/raw_datasets/. Skipping.")
        return

    import json
    rec = None
    with queries_path.open() as f:
        rec = json.loads(f.readline())
    if rec is None:
        print("queries.jsonl is empty.")
        return

    # cocoon's schema: {context, question, answer} (verify against
    # queries_to_parquet.py — the actual key names may differ).
    context = rec.get("context") or rec.get("ctx") or rec.get("repo_text", "")
    question = rec.get("question") or rec.get("query") or ""
    answer = rec.get("answer") or rec.get("response") or ""
    print(f"sample keys: {list(rec.keys())}")
    print(f"context (first 200 chars): {str(context)[:200]!r}")
    print(f"question (first 200 chars): {str(question)[:200]!r}")
    print(f"answer (first 200 chars): {str(answer)[:200]!r}")

    if not (question and answer):
        print("Couldn't pull question/answer from sample; skipping CE compute.")
        return

    # Build a single training-style sequence: question prompt + answer tokens.
    # Mask the prompt tokens from the loss.
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(question)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = prompt + str(answer)
    enc = tokenizer(full, return_tensors="pt").to("cuda")
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
    labels = enc["input_ids"].clone()
    labels[:, :prompt_len] = -100

    model.reset()
    with torch.no_grad():
        out = model.base_model(input_ids=enc["input_ids"], labels=labels)
    print(f"base model CE on this sample (no LoRA): {out.loss.item():.4f}")
    print("[verdict] If this is already ~2 or below, the base model can answer")
    print("          without needing context-conditioned LoRA → no gradient")
    print("          pressure on the hypernet → it stays at zero.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--checks",
        type=str,
        default="1,2,3,4",
        help="Comma-separated list of checks to run (1, 2, 3, 4).",
    )
    args = p.parse_args()
    checks = set(args.checks.split(","))

    ckpt_dir = Path(args.checkpoint)
    weights_path = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (ckpt_dir / name).exists():
            weights_path = ckpt_dir / name
            break
    if weights_path is None:
        sys.exit(f"No model weights found in {ckpt_dir}")
    print(f"[debug] checkpoint: {ckpt_dir}")
    print(f"[debug] weights: {weights_path}")

    saved_args = load_saved_args(ckpt_dir.parent / "args.yaml")
    model, tokenizer = construct_model(saved_args)
    load_weights(model, weights_path)

    if "1" in checks:
        check1_scalers(model)
    if "2" in checks:
        check2_wiring(model, tokenizer)
    if "3" in checks:
        check3_generated_lora_magnitudes(model)
    if "4" in checks:
        check4_base_ce_on_training_sample(model, tokenizer)


if __name__ == "__main__":
    main()
