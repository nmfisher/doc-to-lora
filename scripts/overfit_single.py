"""Architecture diagnostic: can the hypernet learn context-specific LoRAs?

Trains on TWO examples that share the same PROMPT but have different
(context, response) pairs:
  - iterative fibonacci → "O(n) / O(1) / two variables"
  - recursive  fibonacci → "O(2^n) / O(n) stack / no memoization"

The single-example version of this script confirmed the hypernet can
drive train loss to zero, but only by learning a CONSTANT LoRA that
ignored the context (the same response came out for any input context).
Two examples with distinct correct responses force the optimizer to
choose between: (a) learning context-sensitivity, or (b) settling on
some compromise / averaged constant LoRA — which can't drive loss to
zero on both.

Successful run looks like:
  - Both per-example losses drive to ~0.
  - Test outputs differ across contexts (iterative != recursive).
  - Each test output matches its trained response.

Failure mode (architecture-degenerate):
  - Loss stalls around the cross-entropy of mixing the two responses.
  - Test outputs are identical regardless of context.

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
from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.trainer import causal_lm_ce_loss

MODEL = os.environ.get("MODEL_DIR", "google/gemma-4-E2B-it")
TARGET_MODULES = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Two training examples that share the same PROMPT but differ in (context,
# response). A context-blind LoRA can't fit both — it could match the modal
# response or some average, but not produce two distinct correct answers
# when conditioned on different contexts. Successful overfit on BOTH proves
# the architecture can encode context-specific information through the LoRA.
PROMPT = (
    "What is the time complexity of the fibonacci function shown in the "
    "context, and how much auxiliary memory does it use?"
)

CONTEXT_ITER = """\
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

RESPONSE_ITER = (
    "This is the iterative implementation. It uses O(n) time and O(1) "
    "auxiliary memory, storing only two running totals (a and b) and "
    "updating them in a loop. No recursion is involved."
)

CONTEXT_REC = """\
def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number recursively.

    Defines the function in terms of itself: F(n) = F(n-1) + F(n-2).
    There is no memoization, so identical subproblems are recomputed
    many times all the way down to the base case.
    \"\"\"
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

RESPONSE_REC = (
    "This is the naive recursive implementation. It uses O(2^n) time "
    "because identical subproblems are recomputed without memoization, "
    "and O(n) auxiliary stack memory proportional to the recursion depth."
)

EXAMPLES = [
    {"label": "iterative", "context": CONTEXT_ITER, "response": RESPONSE_ITER},
    {"label": "recursive", "context": CONTEXT_REC,  "response": RESPONSE_REC},
]

# Unrelated context used only at test time as a probe — neither trained
# response should appear; we want to see what the model "defaults" to.
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
    # Use the same tokenizer call internalize() uses (train=False default)
    # so train-time and inference-time ctx_ids are byte-for-byte identical.
    ctx_tokenizer = get_tokenizer(MODEL)

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

    section("Step 3: tokenize training examples")
    # Prompt-only (same across examples) — used to find where the response
    # starts for label masking.
    prompt_enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(device)
    prompt_ids = prompt_enc["input_ids"]
    prompt_end = prompt_ids.shape[-1]

    train_examples = []
    for ex in EXAMPLES:
        # IMPORTANT: match what model.internalize() does at inference time.
        # internalize() calls tokenize_ctx_text which WRAPS the context in
        # a chat template ({system, user: <ctx>, add_generation_prompt}).
        # If we tokenize the raw context string here, the hypernet trains
        # on one token distribution and sees a different one at inference,
        # so the LoRA it generates for the same context_str differs between
        # the two paths.
        ctx_ids_list = tokenize_ctx_text(
            dict(context=[ex["context"]]), ctx_tokenizer
        )["ctx_ids"]
        ctx_ids = torch.tensor(ctx_ids_list, device=device)
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
            [{"role": "user", "content": PROMPT},
             {"role": "assistant", "content": ex["response"]}],
            return_tensors="pt",
            add_generation_prompt=False,
            return_dict=True,
        ).to(device)
        full_ids = full_enc["input_ids"]
        attention_mask = torch.ones_like(full_ids)
        labels = full_ids.clone()
        labels[:, :prompt_end] = -100

        n_response_tokens = (labels[0] != -100).sum().item()
        print(f"[overfit] {ex['label']}: ctx_ids {tuple(ctx_ids.shape)}  "
              f"input_ids {tuple(full_ids.shape)}  "
              f"n_response_tokens={n_response_tokens}", flush=True)

        train_examples.append({
            "label": ex["label"],
            "ctx_ids": ctx_ids,
            "ctx_attn_mask": ctx_attn_mask,
            "n_ctx_chunks": n_ctx_chunks,
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    print(f"[overfit] prompt_end={prompt_end} (shared)", flush=True)

    section("Step 4: train")
    optimizer = torch.optim.AdamW(model.hypernet.parameters(), lr=args.lr)
    # The modulated wrapper hardcodes loss=None in its output; we compute
    # CE externally with the same helper the production trainer uses
    # (shift-by-one + cross_entropy, ignore_index=-100).
    vocab_size = tokenizer.vocab_size

    for step in range(args.steps):
        optimizer.zero_grad()
        per_example_losses = []
        # Accumulate gradients across both examples per step (equivalent to
        # gradient accumulation with batch_size=1, accum_steps=len(examples)).
        # Peak memory is one example at a time — the prior example's forward
        # graph is freed when its backward() returns.
        for ex in train_examples:
            outputs, (gen_loras, _) = model(
                ctx_ids=ex["ctx_ids"],
                ctx_attn_mask=ex["ctx_attn_mask"],
                n_ctx_chunks=ex["n_ctx_chunks"],
                input_ids=ex["input_ids"],
                attention_mask=ex["attention_mask"],
                labels=ex["labels"],
                return_generated_lora=True,
            )
            per_token_loss = causal_lm_ce_loss(outputs.logits, ex["labels"], vocab_size)
            n_active = (ex["labels"] != -100).sum().clamp(min=1)
            ce_loss = per_token_loss.sum() / n_active

            if args.l1_reg > 0:
                l1_norm = 0.0
                for module_loras in gen_loras.values():
                    l1_norm += (module_loras["A"].abs().sum(0).mean()
                                + module_loras["B"].abs().sum(0).mean())
                l1 = l1_norm / len(gen_loras)
                loss = ce_loss + args.l1_reg * l1
            else:
                loss = ce_loss
            # Average gradient contribution per example (so total grad scale
            # doesn't depend on how many examples we have).
            (loss / len(train_examples)).backward()
            per_example_losses.append(ce_loss.item())

        optimizer.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            losses_str = "  ".join(
                f"{ex['label']}_ce={l:.4f}"
                for ex, l in zip(train_examples, per_example_losses)
            )
            mean_loss = sum(per_example_losses) / len(per_example_losses)
            print(f"step {step:4d}  {losses_str}  mean={mean_loss:.4f}",
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

    def gen_with_context(ctx_str):
        model.reset()
        model.internalize(ctx_str)
        with torch.no_grad():
            out = model.generate(
                input_ids=test_prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        return tokenizer.decode(out[0, test_prompt_ids.shape[-1]:],
                                skip_special_tokens=True).strip()

    test_outputs = {}
    for ex in EXAMPLES:
        section(f"Test: trained context [{ex['label']}]")
        text = gen_with_context(ex["context"])
        test_outputs[ex["label"]] = text
        print(text)

    section("Test: ALT context (quicksort — neither trained response should appear)")
    text_alt = gen_with_context(CONTEXT_ALT)
    test_outputs["alt"] = text_alt
    print(text_alt)

    section("Test: no LoRA (baseline)")
    model.reset()
    with torch.no_grad():
        out_base = model.base_model.generate(
            input_ids=test_prompt_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    text_base = tokenizer.decode(out_base[0, test_prompt_ids.shape[-1]:],
                                 skip_special_tokens=True).strip()
    print(text_base)

    section("Summary")
    for ex in EXAMPLES:
        print(f"  Target [{ex['label']}]:    {ex['response']!r}")
        print(f"  Got    [{ex['label']}]:    {test_outputs[ex['label']]!r}\n")
    print(f"  ALT context:               {test_outputs['alt']!r}")

    # The key diagnostic question: does the modulated output DIFFER across
    # contexts? A context-blind LoRA gives identical outputs everywhere.
    labels_seen = list(test_outputs.keys())
    iter_out = test_outputs.get("iterative", "")
    rec_out = test_outputs.get("recursive", "")
    print(f"\n  iterative == recursive ?    {iter_out == rec_out}")
    print(f"  iterative contains 'O(n)'   {'O(n)' in iter_out}")
    print(f"  iterative contains 'O(1)'   {'O(1)' in iter_out}")
    print(f"  recursive contains 'O(2^n)' {'O(2^n)' in rec_out}")
    print(f"  recursive contains 'stack'  {'stack' in rec_out.lower()}")


if __name__ == "__main__":
    main()
