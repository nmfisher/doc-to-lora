"""Architecture diagnostic: can the hypernet overfit a single (context, prompt, response)?

We hardcode one example and train the hypernet on it for N steps. If train loss
drives to ~0 AND the modulated model produces the trained response verbatim
when given the trained context, the architecture is capable of context-specific
LoRA generation. If it can't, the failure is structural.

Sanity checks at end:
  - Test 1: trained context + trained prompt -> should produce trained response
  - Test 2: alternative context + trained prompt -> should produce something
    DIFFERENT (proves the response wasn't just memorized from the prompt alone)
  - Test 3: no LoRA at all (baseline) -> the original wrong-generic answer

Run:
    python scripts/overfit_single.py --steps 300 --lr 1e-3

Or starting from an existing checkpoint:
    python scripts/overfit_single.py --checkpoint train_outputs/runs/.../checkpoint-350
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

sys.path.insert(0, "src")

from ctx_to_lora.configs import (
    AggregatorArguments,
    CtxEncoderArguments,
    HypernetArguments,
)
from ctx_to_lora.model_loading import get_lora_config, get_model_and_tokenizer, get_tokenizer
from ctx_to_lora.modeling.ctx_encoder import CTX_ENCODER_TYPE
from ctx_to_lora.modeling.hypernet import (
    ModulatedPretrainedModel,
    get_hypernet_config,
)
from ctx_to_lora.trainer import causal_lm_ce_loss

MODEL = os.environ.get("MODEL_DIR", "google/gemma-4-E2B-it")
TARGET_MODULES = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

CONTEXT = """\
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

PROMPT = (
    "What is the time complexity of the fibonacci function shown in the "
    "context, and how much auxiliary memory does it use?"
)

# Distinctive target answer: contains "O(n)", "O(1)", "(a and b)" — words the
# model won't produce naturally if it falls back to its recursive-Fibonacci prior.
RESPONSE = (
    "The function uses O(n) time and O(1) auxiliary memory. "
    "It only stores two integer variables (a and b) at a time, with no recursion."
)

CONTEXT_ALT = """\
def quicksort(arr):
    \"\"\"Sort using Lomuto partition. Average O(n log n), worst O(n^2).\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[-1]
    less = [x for x in arr[:-1] if x <= pivot]
    greater = [x for x in arr[:-1] if x > pivot]
    return quicksort(less) + [pivot] + quicksort(greater)
"""


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="optional checkpoint dir to load hypernet weights from")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--l1-reg", type=float, default=0.0,
                   help="L1 reg coef on generated LoRAs (0 for overfit; 0.01 matches train)")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Enable gradient checkpointing on base model — cuts "
                   "activation memory ~50% at ~30% slower. Use on <40GB GPUs.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[overfit] device={device}", flush=True)

    section("Step 1: build fresh modulated model")
    peft_config = get_lora_config(
        MODEL, lora_r=8, lora_dropout=0.0,
        target_modules=TARGET_MODULES, lora_alpha=8,
    )
    base_model, tokenizer = get_model_and_tokenizer(
        model_name_or_path=MODEL,
        train=True,
        requires_grad=False,
        use_flash_attn=False,
        peft_config=peft_config,
    )
    ctx_tokenizer = get_tokenizer(MODEL, train=True)

    ctx_cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if hasattr(ctx_cfg, "text_config"):
        ctx_cfg = ctx_cfg.text_config

    ctx_encoder_args = CtxEncoderArguments(
        ctx_encoder_model_name_or_path=MODEL,
        ctx_encoder_type=CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS,
        layer_idx=8,
        ctx_encoder_last_layer=16,
    )
    hypernet_args = HypernetArguments(per_rank_gen=True, per_layer_processing=True)
    aggregator_args = AggregatorArguments(
        n_latent_queries=16, num_blocks=9, num_self_attn_per_block=0,
    )
    hypernet_config = get_hypernet_config(
        base_model, ctx_cfg, hypernet_args, aggregator_args,
        ctx_encoder_args, peft_config=peft_config,
    )

    model = ModulatedPretrainedModel(
        base_model, hypernet_config, ctx_encoder_args,
        use_sequence_packing=False,
    )

    if args.grad_checkpoint:
        # Re-compute base-model activations during backward instead of caching.
        # Cuts ~50% of activation memory for the base forward (the dominant
        # cost here, since we backprop loss -> LoRA -> hypernet, which needs
        # the base activations). ~30% slower per step.
        base_model.gradient_checkpointing_enable()
        # HF docs recommend disabling input grad on the embedding when
        # gradient_checkpointing is on with frozen base — otherwise the
        # checkpoint segment thinks there's nothing to backward through.
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        print("[overfit] gradient checkpointing enabled on base_model",
              flush=True)

    if args.checkpoint:
        section(f"Step 2: load hypernet from {args.checkpoint}")
        ckpt_dir = Path(args.checkpoint)
        weights_path = ckpt_dir / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file
            sd = load_file(str(weights_path), device=device)
        else:
            sd = torch.load(str(ckpt_dir / "pytorch_model.bin"),
                            map_location=device, weights_only=False)
        tensor_only = {k: v for k, v in sd.items() if torch.is_tensor(v)}
        missing, unexpected = model.hypernet.load_state_dict(tensor_only, strict=False)
        print(f"[overfit] loaded {len(tensor_only)} tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    else:
        section("Step 2: starting from fresh hypernet init")

    model.train()

    section("Step 3: tokenize the single training example")
    # Context for the hypernet
    ctx_enc = ctx_tokenizer(CONTEXT, return_tensors="pt").to(device)
    ctx_ids = ctx_enc["input_ids"]
    # Direct model.forward() requires the caller to pass ctx_attn_mask;
    # only the internalize() / generate() paths build it implicitly.
    ctx_attn_mask = torch.ones_like(ctx_ids)
    # n_ctx_chunks[i] = how many chunks context i was split into. Our
    # single context fits in one chunk, so [1]. (split_too_long_ctx in
    # processing.py produces this for the production pipeline.)
    n_ctx_chunks = torch.ones(ctx_ids.shape[0], dtype=torch.int32, device=device)

    # Full chat (user prompt + assistant response). Gemma 4's tokenizer
    # returns a BatchEncoding (dict) from apply_chat_template, so we use
    # return_dict=True and pull ["input_ids"] — same pattern as the
    # working test_gemma4_hypernet.py.
    full_enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE}],
        return_tensors="pt",
        add_generation_prompt=False,
        return_dict=True,
    ).to(device)
    full_ids = full_enc["input_ids"]

    # Prompt-only (to find where the response starts, for label masking)
    prompt_enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(device)
    prompt_ids = prompt_enc["input_ids"]
    prompt_end = prompt_ids.shape[-1]

    input_ids = full_ids
    attention_mask = torch.ones_like(input_ids)
    labels = full_ids.clone()
    labels[:, :prompt_end] = -100

    n_response_tokens = (labels[0] != -100).sum().item()
    print(f"[overfit] ctx_ids {tuple(ctx_ids.shape)}  "
          f"input_ids {tuple(input_ids.shape)}  "
          f"prompt_end={prompt_end}  n_response_tokens={n_response_tokens}", flush=True)

    section("Step 4: train")
    optimizer = torch.optim.AdamW(model.hypernet.parameters(), lr=args.lr)
    # The modulated wrapper hardcodes loss=None in its output; we compute
    # CE externally with the same helper the production trainer uses
    # (shift-by-one + cross_entropy, ignore_index=-100).
    vocab_size = tokenizer.vocab_size

    for step in range(args.steps):
        optimizer.zero_grad()
        outputs, (gen_loras, _) = model(
            ctx_ids=ctx_ids,
            ctx_attn_mask=ctx_attn_mask,
            n_ctx_chunks=n_ctx_chunks,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_generated_lora=True,
        )
        per_token_loss = causal_lm_ce_loss(outputs.logits, labels, vocab_size)
        n_active = (labels != -100).sum().clamp(min=1)
        ce_loss = per_token_loss.sum() / n_active

        if args.l1_reg > 0:
            # Same L1 reg shape as trainer.py: A.abs().sum(0).mean() + B same,
            # averaged over modules.
            l1_norm = 0.0
            for module_loras in gen_loras.values():
                l1_norm += (module_loras["A"].abs().sum(0).mean()
                            + module_loras["B"].abs().sum(0).mean())
            l1 = l1_norm / len(gen_loras)
            total_loss = ce_loss + args.l1_reg * l1
        else:
            l1 = torch.tensor(0.0, device=device)
            total_loss = ce_loss
        total_loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"step {step:4d}  ce={ce_loss.item():.4f}  "
                  f"l1={l1.item():.6f}  total={total_loss.item():.4f}",
                  flush=True)

    model.eval()

    # Build the inference-time prompt (user only, with generation prompt)
    test_prompt_enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(device)
    test_prompt_ids = test_prompt_enc["input_ids"]

    if not getattr(model.ctx_encoder.base_model, "name_or_path", ""):
        model.ctx_encoder.base_model.name_or_path = MODEL

    section("Test 1: trained context + trained prompt (should match RESPONSE)")
    model.reset()
    model.internalize(CONTEXT)
    with torch.no_grad():
        out1 = model.generate(
            input_ids=test_prompt_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    text1 = tokenizer.decode(out1[0, test_prompt_ids.shape[-1]:],
                             skip_special_tokens=True)
    print(text1.strip())

    section("Test 2: ALT context + trained prompt (should differ from Test 1)")
    model.reset()
    model.internalize(CONTEXT_ALT)
    with torch.no_grad():
        out2 = model.generate(
            input_ids=test_prompt_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    text2 = tokenizer.decode(out2[0, test_prompt_ids.shape[-1]:],
                             skip_special_tokens=True)
    print(text2.strip())

    section("Test 3: no LoRA (baseline)")
    model.reset()
    with torch.no_grad():
        out3 = model.base_model.generate(
            input_ids=test_prompt_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    text3 = tokenizer.decode(out3[0, test_prompt_ids.shape[-1]:],
                             skip_special_tokens=True)
    print(text3.strip())

    section("Summary")
    print(f"Target RESPONSE:\n  {RESPONSE!r}")
    print(f"\nTest 1 (trained ctx): {text1.strip()!r}")
    print(f"\nTest 2 (alt ctx):     {text2.strip()!r}")
    print(f"\nIdentical (1 vs 2)?   {text1.strip() == text2.strip()}")
    print(f"Test 1 contains 'O(n)': {'O(n)' in text1}")
    print(f"Test 1 contains 'O(1)': {'O(1)' in text1}")


if __name__ == "__main__":
    main()
