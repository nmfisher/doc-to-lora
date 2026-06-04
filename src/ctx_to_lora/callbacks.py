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

    def __init__(self, tokenizer, max_new_tokens: int = 48):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            _emit("[ctx-probe] skipped: no model in callback kwargs")
            return

        # The Trainer keeps trainable params in train() mode; flip to eval()
        # for deterministic generation, restore at the end. Use no_grad to
        # avoid building the autograd graph for the probe forward.
        was_training = model.training
        model.eval()
        try:
            self._probe(model, state.global_step)
        except Exception as e:
            _emit(f"[ctx-probe] step={state.global_step} FAILED: "
                  f"{type(e).__name__}: {e}")
        finally:
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
