# Plan: Integrate ChunkFT into doc-to-lora

## Context

**What we train.** The base model (Gemma-3/4) and `ctx_encoder` are frozen (`train.py:258-261`). The only trainee is the `HyperLoRA` hypernet inside `ModulatedPretrainedModel` (`src/ctx_to_lora/modeling/hypernet.py`). At forward time the hypernet emits per-context LoRA A/B matrices that are patched into the base model's linear layers via `lora_forward_packed` (`src/ctx_to_lora/modeling/lora_layer.py`).

**Where the memory goes.** The hypernet's `head.weight` shape `[n_layers, n_modules, d_latent, d_lora]` dominates trainable params — for Gemma-4 it's roughly 1.3 B params in a single tensor, so Adam states alone are ~10 GB fp32. The pre-head `ResMLPBlock` stack and `bias_A/B/scaler_A/B` ParameterDicts are small in comparison. The frozen base model's forward activations (packed_inp_len=16 384 × hidden × ~35 layers, with gradient checkpointing already on) are likely a larger contributor to peak VRAM than optimizer states.

**Why chunk anyway.** Per user direction, we build the chunk integration with `chunk_tuning=False` as the default off-switch. That way the codepath is available for experiments and A/B comparison without changing default behavior. Convergence trade-off (block coordinate descent over `n_layers`) is accepted as a research question and surfaced via a divergence check.

**What ChunkFT does** (from https://github.com/misonsky/chunk). Partitions trainable params along a chunk dim, sets `requires_grad=True` only on the active chunk, parks inactive chunks' optimizer states on CPU, rotates to the next chunk every `chunk_update_interval` optimizer steps, and steps the LR scheduler once per full cycle.

## Recommended Approach

A vendored, minimal reimplementation under `src/ctx_to_lora/chunk/`, composed into our existing trainer chain via a mixin. No git submodule, no pip dep. Approximately three files, all toggleable.

### 1. New module `src/ctx_to_lora/chunk/`

**`chunk_handler.py`** — owns per-param chunk metadata and the active-chunk toggle.
- `ChunkHandler(model, trainable_param_names, chunk_num, strategy)` builds a list of `(param, ranges)` records for every name in `trainable_param_names`.
- Chunking rules:
  - `head.weight` (4D, `[n_layers, n_modules, d_latent, d_lora]`): chunk along dim 0 (`n_layers`). `chunk_num` is clamped to ≤ `n_layers`.
  - `ResMLPBlock` Linear weights (2D): chunk along `strategy` (row/col).
  - `bias_A/B/scaler_A/B` (small ParameterDicts): never chunked, always trainable.
  - Aggregator weights: chunk along dim 0 if 2D, otherwise stay always-trainable.
- `set_active_chunk(idx)`: sets `requires_grad=True` for the active slice (mask-based: zero out non-active gradient contributions in a backward hook) and `False` for the others. Note: PyTorch doesn't let `requires_grad` vary across slices of one Tensor — implemented via a `register_post_accumulate_grad_hook` that masks grad to zero outside the active range.
- `advance()`: increments counter modulo `chunk_num` and returns whether a full cycle just completed.

**`cpu_offload_optimizer.py`** — thin wrapper around `torch.optim.AdamW`.
- Maintains per-param pinned CPU copies of `exp_avg`, `exp_avg_sq` keyed by chunk index.
- On `step()`: copies active-chunk states GPU↔CPU lazily; calls underlying `AdamW.step()`.
- Implements `prefetch_chunk_states(parameters, target_counter)` so ChunkTrainer's prefetch hook works.

**`trainer_mixin.py`** — `ChunkTrainerMixin` providing the HF Trainer overrides:
- `create_optimizer`: builds parameter groups filtered to `trainable_param_names`, preserves the monkey-patched `get_decay_parameter_names` (`trainer.py:450`), instantiates `CpuOffloadAdamW`.
- `training_step`: after the wrapped step, calls `chunk_handler.advance()`; if full cycle completed, allows LR scheduler to step.
- `_save_checkpoint` / `_load_checkpoint`: serialize chunk state to `chunkft_state.json` alongside the existing checkpoint.
- `clip_active_chunk_grads`: clips only active chunks' grads.
- All overrides are no-op when `chunk_tuning=False`.

### 2. Trainer composition (`src/ctx_to_lora/trainer.py`)

Change inheritance:
```
class CrossEntropyTrainer(ChunkTrainerMixin, ModulatedModelTrainer):
class DistillationTrainer(ChunkTrainerMixin, ModulatedModelTrainer):
```
Mixin is leftmost so its overrides win. `__init__` accepts and stores chunk args, then defers to super.

Update `train_model` (`trainer.py:400`) to pop chunk fields off `training_args` and pass them into `trainer_kwargs`.

### 3. Config & CLI (`src/ctx_to_lora/configs.py`)

Add `ChunkArguments` dataclass parallel to `HypernetArguments`:
- `chunk_tuning: bool = False`
- `chunk_num: int = 4`
- `chunk_strategy: Literal["row","col"] = "row"`
- `chunk_update_interval: int = 500`
- `enable_chunk_prefetch: bool = True`

Register in the `ArgumentParser` tuple in `train.py:74-85` and in `validate_args` (`train.py:91`). Add to `add_safe_globals` if present.

### 4. torch.compile interaction (`train.py:269`)

`model.hypernet.compile(fullgraph=True, mode="max-autotune")` is incompatible with per-step `requires_grad` toggling — Dynamo recompiles every rotation. When `chunk_tuning=True`, downgrade to `model.hypernet.compile()` (no fullgraph, default mode) or skip compile entirely. Gate via the config flag.

### 5. Callback guard (`src/ctx_to_lora/callbacks.py`)

`CtxSensitivityProbe` should not alarm before the first full chunk cycle completes (most head params haven't trained yet). Add `min_steps_before_alarm` honored when `chunk_tuning=True`, set to `chunk_num × chunk_update_interval` at construction time in `train.py`.

### 6. Loss/regularizer (no change needed)

`gen_lora_l1_reg` (`trainer.py:209-218, 350-359`) penalizes the *output* of the head, not the head weights. The penalty value is unchanged by chunking; only its gradient routing is sparser. Document this in `ChunkArguments` help text — no coefficient scaling.

## Files To Modify / Create

- **Create:**
  - `src/ctx_to_lora/chunk/__init__.py`
  - `src/ctx_to_lora/chunk/chunk_handler.py`
  - `src/ctx_to_lora/chunk/cpu_offload_optimizer.py`
  - `src/ctx_to_lora/chunk/trainer_mixin.py`
- **Modify:**
  - `src/ctx_to_lora/configs.py` — add `ChunkArguments`
  - `src/ctx_to_lora/trainer.py` — mixin into `CrossEntropyTrainer`/`DistillationTrainer`, plumb args through `train_model`
  - `train.py` — register `ChunkArguments`, conditionally weaken compile, configure callback
  - `src/ctx_to_lora/callbacks.py` — add `min_steps_before_alarm` knob
  - one example config under `configs/main_exp/` — add commented `chunk_tuning` block

## Pre-implementation Reads

Before writing the mixin and offload optimizer, read the actual source (the WebFetch summary is not enough to ship from):
- `gh api repos/misonsky/chunk/contents/chnk/trainer.py` — full `_inner_training_loop`, exact LR-scheduler gating.
- `gh api repos/misonsky/chunk/contents/optimizers/optimization.py` (and any sibling files) — confirm `prefetch_chunk_states` signature and the CPU/GPU copy pattern.
- `src/ctx_to_lora/modeling/lora_layer.py` — `lora_forward_packed` signature, since the chunk path doesn't touch it but we want to confirm no interaction with grad masking.
- `src/ctx_to_lora/tracker/cuda_memory_tracker.py` — use existing instrumentation in the verification phase.

## Verification

1. **Unit test** in `scripts/` — instantiate hypernet via an `examples/` config, build `ChunkHandler(chunk_num=4)`, assert sum of chunk numels equals total numel and exactly one chunk is "active" at a time.
2. **Memory baseline** with the existing `cuda_memory_tracker` — one batch, three configs: baseline, `chunk_tuning=True chunk_num=4`, `chunk_tuning=True chunk_num=8`. Record peak VRAM split by hypernet fwd, base fwd, optimizer step.
3. **Divergence check via `overfit_single.py`** (added in commit e57491a) — run baseline for N steps, save loss curve. Rerun with `chunk_tuning=True chunk_num=4 chunk_update_interval=10`; loss at step `4·N` should be within ~5% of baseline. Catches incorrect grad routing or stale optimizer state.
4. **Checkpoint round-trip** — `chunk_tuning=True` smoke run, kill mid-cycle, resume from checkpoint, confirm `chunkft_state.json` restores chunk counter and that the next step activates the correct chunk.
5. **Real-config 1 000-step run** — `gemma4_e2b_closed_qa.yaml` with chunk on; wandb metrics for grad_norm, eval loss, `CtxSensitivityProbe` health.

## Caveats

- Pinned-memory CPU copies for chunk optimizer states require ~10 GB host RAM for Gemma-4 head's Adam states. Verify host capacity before recommending in cloud configs.
- BCD-style training may interact with `gen_lora_l1_reg` over a full cycle in non-obvious ways (the L1 path's gradient sees only the active chunk per step). Watch eval-loss vs train-loss divergence in the 1 000-step run.
- Recompile cost when toggling `requires_grad` is unmeasured — the plan recommends weakening compile under `chunk_tuning=True`. If we instead keep compile on, measure step-time impact with `TORCH_LOGS=recompiles`.
