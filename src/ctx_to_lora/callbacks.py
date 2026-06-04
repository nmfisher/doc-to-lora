"""Trainer callbacks for in-training diagnostics.

CtxSensitivityProbe fires on every checkpoint save and runs a 3-way
generation comparison: iterative fibonacci / recursive fibonacci /
quicksort, all asked the same prompt. Logs both to stdout (visible
between training progress bars) and to the run's debug.log via the
package logger.

This is the smoke alarm for the context-blind LoRA failure mode that
let v4 / v6 run for hundreds of steps with falling loss but byte-
identical generations across contexts. With the probe wired up,
that mode is visible by step 5 of any run.
"""

import logging

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


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


def _emit(msg: str) -> None:
    """Print to stdout (visible in live training log) AND to the package
    logger (captured in the run's debug.log). flush=True so it shows up
    immediately between tqdm progress redraws, not buffered at exit."""
    print(msg, flush=True)
    logger.info(msg)


class CtxSensitivityProbe(TrainerCallback):
    """Fires on every checkpoint save. Generates a fixed prompt against three
    distinct contexts and reports whether the outputs differ. A run where all
    three outputs collapse to the same string is producing a context-blind
    LoRA — that's the failure mode this probe exists to catch.

    Args:
        tokenizer: The base-model tokenizer (used to encode the prompt and
            decode generated tokens).
        max_new_tokens: Generation length per context. 48 is enough to see
            whether the time-complexity answer is right without burning
            seconds on a long generation.
    """

    def __init__(self, tokenizer, model_name_or_path: str,
                 max_new_tokens: int = 48):
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            _emit("[ctx-probe] skipped: no model in callback kwargs")
            return

        # internalize() loads a tokenizer via ctx_encoder.base_model.name_or_path.
        # train.py doesn't set name_or_path explicitly, and the wrapped HF model
        # often has it empty by the time we're here — without this, internalize()
        # raises OSError("Repo id... '': ..."). Match overfit_single.py:354.
        ctx_base = getattr(model, "ctx_encoder", None)
        if ctx_base is not None and not getattr(ctx_base.base_model,
                                                "name_or_path", ""):
            ctx_base.base_model.name_or_path = self.model_name_or_path

        # Training patches each LoRA-targeted forward with lora_forward_packed,
        # which expects seq_lens/tot_len from the data collator. The probe's
        # model.generate() doesn't supply those, so the packed forward crashes
        # with `TypeError: lora_forward_packed() missing 2 required positional
        # arguments`. Flip the flag, force a re-patch with the unpacked
        # lora_forward for the probe, restore for training in finally.
        was_training = model.training
        original_packing = getattr(model, "use_sequence_packing", True)
        model.eval()
        try:
            if hasattr(model, "patch_lora_forward"):
                model.use_sequence_packing = False
                model.reset()  # clear patched_forward flags
                model.patch_lora_forward()  # re-patch with non-packed forward
            self._probe(model, state.global_step)
        except Exception as e:
            _emit(f"[ctx-probe] step={state.global_step} FAILED: "
                  f"{type(e).__name__}: {e}")
        finally:
            # Restore everything the probe could have disturbed:
            # 1. Sequence packing mode + re-patch with the packed forward.
            #    Without this the next training step's wrapper passes n_qs to
            #    plain Linear.forward (TypeError: 'n_qs' unexpected), or hits
            #    the wrong forward variant.
            if hasattr(model, "patch_lora_forward"):
                model.use_sequence_packing = original_packing
                try:
                    model.reset()
                    model.patch_lora_forward()
                except Exception as e:
                    _emit(f"[ctx-probe] step={state.global_step} "
                          f"REPATCH FAILED: {type(e).__name__}: {e}")
            # 2. Drop any LoRA the probe generated so the next train step
            #    starts from a clean state. (No-op if probe already reset.)
            if hasattr(model, "generated_loras"):
                model.generated_loras = None
            # 3. Restore train mode if we toggled.
            if was_training:
                model.train()

    @torch.no_grad()
    def _probe(self, model, step: int) -> None:
        device = next(model.parameters()).device

        prompt_enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        prompt_ids = prompt_enc["input_ids"].to(device)
        prompt_len = prompt_ids.shape[-1]

        outputs = {}
        for label, ctx_text in CONTEXTS.items():
            # reset() clears any cached LoRA so the new internalize() builds
            # a fresh one — without it the second context would inherit the
            # first context's LoRA and the probe would falsely report
            # "context-sensitive" via stale state.
            model.reset()
            model.internalize(ctx_text)
            out = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            text = self.tokenizer.decode(
                out[0, prompt_len:], skip_special_tokens=True,
            ).strip()
            outputs[label] = text

        # Clear LoRA state so it doesn't leak into the next training step.
        model.reset()

        distinct = len(set(outputs.values()))
        header = (
            f"[ctx-probe] step={step} distinct_outputs={distinct}/3 "
            f"(want=3 for context-sensitive)"
        )
        _emit(header)
        for label, text in outputs.items():
            _emit(f"[ctx-probe] step={step} {label}: {text!r}")
