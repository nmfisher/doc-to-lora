# Plan: Add Gemma 4 Support to Doc-to-LoRA (v2.1)

## Changes from v1 (after reviewing the codebase + the text-to-lora precedent + Gemma 4 docs)

- **New Phase 0**: the transformers upgrade is a hard blocker (we're pinned at `4.51.3`, which has no `gemma4` module), not a closing footnote.
- **Phase 2 expanded**: PEFT is *not* only needed for init. The runtime LoRA-injection machinery (`patch_lora_forward`, `reset`) also keys off PEFT `BaseTunerLayer` instances. Those must be reworked too.
- **Phase 3 expanded**: layer filtering yields **non-contiguous** indices — a cross-cutting invariant that must be threaded through training, checkpoint save/load, and eval. There are ~4 dimension profiles (not a clean local/global 2-way split).
- **New Phase 6**: eval/checkpoint-load re-introduces PEFT (`PeftModel.from_pretrained`) and re-introduces the vision/audio-tower contamination problem.
- **Phase 5 corrected**: real control tokens are **split** (`<|turn>`…`<turn|>`), not `<|turn|>`. Do not hand-roll a chat template.
- **Context section corrected**: the precedent was ~60 commits, and its cited endpoint is itself a plan doc, not implementation.
- Resolved the v1 self-contradiction on the load class (see Phase 1 + Verification).
- **v2.1**: use Gemma 4 as **both** the context encoder and the base model (see Scope) — Phase 4 is now first-class, and the two roles share one tokenizer.

## Scope decision: Gemma 4 as BOTH context encoder and base model

Gemma 4 fills **both roles** — base model (LoRA target) *and* context encoder (reads the document). Implications:

- **Two instances of the same checkpoint, not one shared object.** The base model's forward is monkey-patched for LoRA injection and stays modulated; the encoder stays frozen and clean. The architecture already keeps `model.base_model` and `model.ctx_encoder` separate — point both at `google/gemma-4-E2B-it`.
- **Encoder runs text-only by default**: extract `.language_model` (same as the base) and early-exit at `num_hidden_layers // 4`. This gives architectural parity and the one real win of sharing a model — a **single shared tokenizer** for both sides (no cross-tokenizer reconciliation in the data pipeline).
- **Multimodal context is a future opt-in, not this pass.** Using Gemma 4's vision/audio towers on the encoder (the paper's "VLM-encoder → text-base" trick, extended to audio) means *keeping* the towers instead of extracting `.language_model`, plus an image/audio data pipeline. Designed-for, but out of scope for v1.

Net effect: **both the base-model path (Phases 1–3, 6) and the encoder path (Phase 4) are in scope.** Phase 4 is no longer conditional.

**Memory**: Gemma 4 E2B is ~5.1B full params (Per-Layer Embeddings inflate the count past the "effective 2B"), so two instances roughly double base memory. The encoder is early-exit, so truncating it to the first `num_hidden_layers // 4 + 1` layers is a valid optimization — check whether the precedent did this.

## Probe results (transformers 5.9.0, `DEVICE=meta`, 2026-05-29)

Ran `scripts/probe_gemma4.py` against `google/gemma-4-E2B-it`. **Confirmed**: 35 layers (only via `text_config.num_hidden_layers`; no top-level), 4 dimension profiles, majority group **16/35** = `[15,16,17,18,20,21,22,23,25,26,27,28,30,31,32,33]` (non-contiguous), split control tokens (`<|turn>`=105/`<turn|>`=106, `<|tool_call>`=48/49, `<|tool>`=46/47), `hidden_size`=1536, vocab 262144, ctx 131072, vision + audio towers present.

**Empirical corrections to the plan:**
1. **Extraction path is `.model.language_model`** (→ `Gemma4TextModel`) — one level deeper than Gemma 3. Fixes Phase 1.2.
2. **Decoder targets are plain `nn.Linear` with accurate `.in_features`/`.out_features`.** `Gemma4ClippableLinear` lives only in the vision/audio towers, not the decoder → the `weight.shape` workaround (2.1) and own-forward workaround (2.5) are **defensive-only, not required**; `nn.Linear.forward` patching is safe. Phase 2 is simpler than feared.
3. **The 16-layer majority group has NO `k_proj`/`v_proj`** (unified/shared KV). `target_modules` there = **`q_proj, o_proj, gate_proj, up_proj, down_proj`** — drop k/v. This is the "filter to modules present" case (3.3); k/v exist only in the smaller (6144-MLP) profiles.
4. **MLP width varies** (`gate/up` 6144 vs 12288) — MLP-only does not escape filtering.
5. **>50% of layers dropped**: majority group adapts only 16 of 35 (drops `[0–14, 19, 24, 29, 34]`) — a real modeling restriction to weigh.
6. **`<|think|>` exists** (id 98) — reasoning token is real, not speculative (5.2). Gemma 2/3 `<start_of_turn>`/`<end_of_turn>` are gone.

**Step 7 (CPU run) confirmed**: residual width uniform = 1536 across all 36 hidden states, output exposes `.last_hidden_state`, early-exit layer `35//4=8` → `(1, seq, 1536)`. Stored weights are a single 10.2 GB bf16 `model.safetensors` (≈ the bf16 VRAM footprint). Empirical CTX_AFFIXES derived — see Phase 5.1.

---

## Context

Add support for Google's Gemma 4 models (starting `google/gemma-4-E2B-it`, released 2026-04-02) to the Doc-to-LoRA hypernetwork training pipeline. Build a general solution extending to E4B, 26B-A4B, and 31B.

This was previously done in the text-to-lora repo (`/Volumes/T7/projects/text-to-lora`). Useful precedent, but note:
- It was **~60 commits** (`6e8ded9..ad58af4`), not 24, and most were a debugging tail — the hard lessons live there.
- The cited endpoint `ad58af4` is just `add GEMMA4_v2.md` (a plan doc). The first commit `6e8ded9` shipped a chat template + token format that were **both wrong and deleted two commits later**. The endpoints are the least representative commits in the range.
- **The two repos have diverged and renamed things.** text-to-lora's `HyperModulator` is doc-to-lora's `ModulatedPretrainedModel`; `get_in_out_features` → `get_peft_in_out_features`; `create_hypermod` → `get_hypernet_config`. **You cannot copy-paste** — every port needs translation. (doc-to-lora's own names, used throughout this plan, are all verified to exist.)

**Why Gemma 4 is hard**: multimodal `Gemma4ForConditionalGeneration` with vision + USM audio towers wrapping a `language_model`. The towers use `Gemma4ClippableLinear` (not `nn.Linear`), so PEFT can't wrap the full model. Composite config, no `use_cache`, and a heterogeneous decoder stack (interleaved local sliding-window vs global attention → different projection dims per layer).

---

## Phase 0: Dependency upgrade (BLOCKER — do first)

### 0.1 Bump transformers — DONE (staged, not installed)
- Was `transformers==4.51.3` (hard-pinned, no `gemma4` module). Bumped `pyproject.toml:9` → `transformers>=5.5.0`; `uv lock` re-resolves to **transformers 5.9.0** (newest under this repo's `exclude-newer = 2026-05-24` ceiling). **Verified in an isolated env that 5.9.0 ships `transformers.models.gemma4`** → the probe's Step 0 import passes once installed.
- **Cascade**: hf-hub 0.35 → **1.16 (major)**, `tokenizers` 0.22.2, `regex`/`typer`/`hf-xet` bumped. `vllm`/`deepspeed`/`accelerate` unchanged in the lock.
- **vllm caveat**: `vllm==0.8.5.post1` produced *no declared* conflict, so the lock resolves — but uv checks declared constraints, **not runtime compat**. vllm 0.8.5 predates transformers 5.x and may break at import/runtime. The precedent removed vllm from core deps (eval-only); re-check in Phase 6.
- **Not installed on this machine**: macOS + `vllm`/`deepspeed`/`liger-kernel` are Linux/CUDA — run `uv sync` + the probe on the GPU box. (A minimal `transformers`+`torch`+`accelerate` env suffices for a CPU-only probe.)
- Minor: hf-hub 1.x dropped the `hf-transfer` extra, so `huggingface-hub[hf-transfer]` (`pyproject.toml:34`) now warns; harmless (`hf_transfer` is a direct dep too) but worth cleaning up.
- `peft` still unpinned (`pyproject.toml:14`) — resolved fine; verify behavior against transformers 5.x during Phase 2.

### 0.2 Regression-guard the existing models
- A 5.x bump risks breaking the existing PEFT / Gemma-2 / Gemma-3 paths. **Before** touching Gemma 4, run an existing config end-to-end (load → train a few steps → eval) on `google/gemma-2-2b-it` to confirm the bump didn't regress current behavior.
- Verify `Gemma3ForConditionalGeneration` (imported at `model_loading.py:13`) still imports and that `from transformers import Gemma4ForConditionalGeneration` now resolves.

---

## Phase 1: Model Loading (`model_loading.py`)

### 1.1 Register Gemma 4 as multimodal
- Add `"google/gemma-4-E2B-it"` (+ E4B / 26B-A4B / 31B) to `GEMMA_VISION_MODELS` (`model_loading.py:18-22`; consumed by `check_is_vision_model()` at `:25-26`). Consider renaming to `MULTIMODAL_MODELS` since Gemma 4 adds audio.
- Import `Gemma4ForConditionalGeneration` (guard for older transformers if you keep any back-compat).

### 1.2 Load + extract `.language_model` (pick ONE strategy)
- Mirror the existing Gemma 3 path (`model_loading.py:157-158`): load with `Gemma4ForConditionalGeneration.from_pretrained(...)`, then `model = model.model.language_model` — **the probe found Gemma 4 nests one level deeper than Gemma 3** (result is a `Gemma4TextModel`).
- **Resolves v1 contradiction**: do **not** use `AutoModelForCausalLM` for Gemma 4 in this repo — vision models go through the `Gemma*ForConditionalGeneration` branch. (text-to-lora used `AutoModelForCausalLM` + `trust_remote_code` auto-dispatch — a valid *alternative*, but don't mix the two.)
- `use_cache` is already popped for vision models (`model_loading.py:128-131`).
- Skip flash-attention (use `"eager"` or default).

### 1.3 Skip PEFT wrapping for Gemma 4
- PEFT wrap happens at `model_loading.py:159-160`: `if peft_config is not None: model = PeftModel(model, peft_config)` (note: `PeftModel(...)`, **not** `get_peft_model`).
- For Gemma 4, gate this off (the precedent uses a `"gemma-4" not in model_path.lower()` string check) and return the bare extracted `language_model`.
- **New plumbing**: `get_model()` has no parameter today to mean "skip PEFT but remember the config." Add one, and thread `peft_config` to the caller so the hypernetwork can receive it directly (see 2.4).

### 1.4 Chat template — already handled
- `get_tokenizer()` checks `os.path.exists(template_path)` and falls back to the tokenizer's built-in template (`model_loading.py:81-86`). Gemma 4 ships a native template → no custom `.jinja`. **Do not write one** (the precedent wasted a commit doing so).

### 1.5 Composite config
- `num_hidden_layers` lives at `config.text_config.num_hidden_layers`. Touch points:
  - `train.py:169-172` — already routes vision models to `.text_config`. ✅
  - `train.py:189-192` — `ctx_encoder_args.layer_idx = num_hidden_layers // 4` (ctx-encoder side; **now in scope** — read from `text_config`, see Phase 4).
  - `train.py:203/206` — `base_model.config.num_hidden_layers` (note: assignment is at `:206`, not `:205`).
  - `train.py:262` — `model.ctx_encoder.config.max_position_embeddings`.
  - `train.py:278` — `model.base_model.config.max_position_embeddings` (**missed in v1**).

---

## Phase 2: Non-PEFT LoRA path

> v1's framing — "PEFT is only needed for initialization" — is **wrong**. The runtime injection attaches to PEFT `BaseTunerLayer` instances. Both init *and* the forward-patching machinery must get non-PEFT branches.

### 2.1 Dimension reading — `get_peft_in_out_features()` (`utils.py:174`)
- Returns `(None, None)` when `peft_config is None` (`:178-179`); reads `module.in_features` / `module.out_features` at `:196-197`.
- **New**: when PEFT is skipped, iterate the layer's modules matching `target_modules` and read dims from `module.weight.shape` (`[out, in]`, detect via `weight.dim()==2`). `Gemma4ClippableLinear` can report incorrect `.in_features`/`.out_features` — `weight.shape` is authoritative. **Verify** whether the decoder's target modules are actually `Gemma4ClippableLinear` or plain `nn.Linear` (the class is pervasive in the *towers*; confirm for the language model). Use a reference layer from the **filtered** set (Phase 3), not layer 0.

### 2.2 Module discovery — `get_peft_modules()` (`utils.py:164`)
- Filters on `isinstance(module, BaseTunerLayer)` + `check_target_module_exists(...)` (`:169-170`).
- **New**: non-PEFT branch iterates `model.named_modules()` scoped to the decoder layers, matches `target_modules`, returns the `nn.Linear` modules directly.

### 2.3 Init weights — `get_init_peft_weights()` (`hypernet.py:117`)
- Reads LoRA init from PEFT `BaseTunerLayer` submodules (`:125`, `:137-145`).
- **New**: construct manually — `lora_A` kaiming-uniform (`a=sqrt(5)`), `lora_B` zeros, rank `r` — matching PEFT defaults.

### 2.4 Hypernet config + model wiring
- `get_hypernet_config()` (`hypernet.py:83`): already does `getattr(model, "peft_config", None)` (`:91`) **but then unconditionally reads `lora_config.r` at `:110`** → crashes when None. Thread `peft_config` in from the caller and guard `:110`.
- `ModulatedPretrainedModel.__init__` (`hypernet.py:441`): `self.peft_config = base_model.peft_config["default"]` (`:455`) fails with no PEFT. Accept `peft_config` as a kwarg and store it directly.
- `train.py:358-362` (`isinstance(model.base_model, PeftModel)`) **already has an `else` branch** — half-done, just confirm it covers the bare-model case.

### 2.5 Runtime injection (the part v1 omitted)
- `ModulatedPretrainedModel.patch_lora_forward()` (`hypernet.py:501-521`) and `reset()` (`hypernet.py:804-813`) iterate `get_peft_modules(...)` and replace `module.forward` on PEFT tuner layers.
- `lora_layer.py` forwards call `torch.nn.Linear.forward(self, x)` (`:31` and `:55`), bound to PEFT's `lora.Linear`.
- **New**: without PEFT the targets are plain `nn.Linear` (post-`.language_model` extraction). Re-point patching at those modules and ensure the patched forward calls the correct base forward. **Risk**: if a target is `Gemma4ClippableLinear` (not an `nn.Linear` subclass), `nn.Linear.forward(self, x)` is invalid — call the module's own forward instead.

### 2.6 Module-name list — `get_lora_module_names()` (`utils.py:229`)
- Uses `get_peft_model_state_dict(model)` (`:238`); sizes list with `len(layer_indices)` (`:235`); indexes by **raw** `layer_idx` (`:245`).
- **New**: build PEFT-style key strings (`base_model.model.{name}.lora_A.default.weight`) from `named_modules()` scoped to the decoder; size with `max(layer_indices)+1` (non-contiguous — see Phase 3).

---

## Phase 3: Heterogeneous layers + non-contiguous indices

### 3.1 The real shape of the problem
- Gemma 4 interleaves local sliding-window and global attention (`head_dim=256` local vs `global_head_dim=512` global; final layer always global). Projection dims differ per layer.
- **It is not a clean 2-way split.** The precedent found **~4 distinct dimension profiles** across 35 layers; the early "28 local / 7 global" framing was wrong. The majority profile was only **~16 of 35 layers**.

### 3.2 Filtering strategy (final precedent mechanism)
- For each layer, build a profile = tuple of `(d_in, d_out)` per target module (dims from `weight.shape`). Group layers by profile; pick the **largest group**; that set is `layer_indices`. Don't anchor to layer 0.

### 3.3 Non-contiguous indices are a cross-cutting invariant (v1 under-weighted this)
- The chosen `layer_indices` are **non-contiguous** (e.g. `[0,1,2,5,6,...]`). In the precedent this broke a cascade:
  - List sizing: `max(layer_indices)+1`, never `len(...)` (see 2.6).
  - Use **enumerate-position** for array slicing, **raw index** only for module-name lookup.
  - Some local layers **lack** projections others have → filter `target_modules` to those actually present per layer, or hit `KeyError`.
- **Thread the *same* `layer_indices` through training, checkpoint save/load, and eval.** Persist it in the checkpoint (Phase 6) — recomputing must be deterministic or LoRA maps onto the wrong layers.

### 3.4 Module path resolution — `lora_layer.py:96-100`
- Hardcoded `self_attn.{mname}` / `mlp.{mname}` with **no `else`** → undefined `long_mname` → `NameError` for any module name outside the two lists. Standard targets (`q/k/v/o_proj`, `down/gate/up_proj`) work, but add an explicit `else` raising a clear error. (v1 rated this "low / may work as-is"; it's a hard crash for anything non-standard, not a silent fallback.)

---

## Phase 4: Context encoder (now first-class — Gemma 4 fills this role too)

### 4.1 Loading — same route as the base
- The encoder loads via the Phase 1 path: `Gemma4ForConditionalGeneration` → extract `.language_model`. `check_is_vision_model()` keys on the model name, so registering Gemma 4 (1.1) covers the encoder automatically.
- Composite-config touch points on the encoder side are now live: `train.py:169-172` (already routes to `.text_config` ✅), `:189-192` (`layer_idx = text_config.num_hidden_layers // 4`), `:262` (`ctx_encoder.config.max_position_embeddings` — may need `.text_config`).

### 4.2 `get_layers()` (`utils.py:41-44`)
- Recurses `model.model` until `.layers`. After `.language_model` extraction the stack is standard; verify it resolves (the precedent added a `language_model` branch for the unextracted case).

### 4.3 Early-exit on a heterogeneous stack
- The encoder taps `.last_hidden_state` at the early-exit layer. The **residual-stream width (`hidden_size`) is uniform across all layers** — the local/global heterogeneity is in attention head dims, not the residual stream — so early-exit extraction is dimensionally safe wherever the exit layer lands. Confirm output is `BaseModelOutputWithPast` with `.last_hidden_state` after extraction.

### 4.4 Per-Layer Embeddings (PLE) — verify
- Gemma 4 adds a PLE conditioning vector at every decoder layer. Likely orthogonal to LoRA, but it shapes per-layer activations — confirm the hypernetwork reads the representation you intend (and that the early-exit hidden state is post-PLE as expected, not a mismatched tap).

---

## Phase 5: Data pipeline

### 5.1 CTX_AFFIXES (`data/definitions.py:8-23`)
- Add a `"google/gemma-4-E2B-it"` entry. Existing entries (`gemma-2-2b-it`, `Mistral-7B-Instruct-v0.2`, `Qwen3-4B-Instruct-2507`) are dicts of `"prefix"`/`"suffix"` → **lists of int token IDs**. No Gemma 3 entry exists, so this follows the pattern.
- Compute IDs **empirically** by running the tokenizer (262K vocab → IDs differ from Gemma 2).
- **Simplification from sharing one model**: a single tokenizer serves both encoder and base, so you need exactly one CTX_AFFIXES entry and one set of control-token IDs.
- **Derived from the probe's chat-template token stream** (`<bos><|turn>user\n … <turn|>\n<|turn>model\n`):
  ```python
  "google/gemma-4-E2B-it": {"prefix": [2, 105, 2364, 107], "suffix": [106, 107, 105, 4368, 107]},
  ```
  prefix = `<bos> <|turn> user \n`; suffix = `<turn|> \n <|turn> model \n` — same structure as the gemma-2 entry, with Gemma 4 IDs. (IDs are authoritative from the tokenizer; `2364`/`4368` are the `user`/`model` role tokens.)

### 5.2 Control tokens are SPLIT (v1 had these wrong)
- Real format: `<|turn>`…`<turn|>`, `<|tool_call>`…`<tool_call|>`, `<|tool>`…`<tool|>` — **not** `<|turn|>` / `<|end_turn|>`. The precedent guessed the single-token form and deleted it.
- The `<|think|>` / reasoning handling in v1 is **unverified** — I found tool-call tokens in the precedent, not think tokens. Confirm against the real tokenizer before building thinking-strip logic into data gen or eval.

### 5.3 Chat template
- Use the native `tokenizer.apply_chat_template()`. Do **not** author a custom `.jinja` (see 1.4).

---

## Phase 6: Evaluation + checkpointing (NEW — absent from v1)

### 6.1 Eval re-introduces PEFT
- Even though training bypasses PEFT, eval/checkpoint-load uses `PeftModel.from_pretrained` to load the saved LoRA. That re-introduces every PEFT assumption — and the **vision/audio towers**.
- The precedent took **3 iterations** to stop PEFT matching `Gemma4ClippableLinear` in the towers. Final fix: override `target_modules` with a regex anchored to the full dotted path PEFT matches via `re.fullmatch`, **including the leading `model\.`**:
  `model\.language_model\.layers\.\d+\.(?:self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|down_proj|gate_proj|up_proj)`
- Relevant eval path: `run_eval.py` → `eval_utils.py:evaluate()` (`:703`, `Seq2SeqTrainer`, `predict_with_generate`); generation via `ModulatedPretrainedModel.generate()` (`hypernet.py:815-915`).

### 6.2 Persist + reproduce `layer_indices`
- Save the filtered `layer_indices` in the checkpoint and reload it at eval — do not recompute and hope it matches (Phase 3.3).

---

## Phase 7: Training config + launch

### 7.1 Config
- **New file**: `configs/main_exp/gemma4_e2b_closed_qa.yaml`. Point **both** the base (`model_name_or_path`) **and** `ctx_encoder_model_name_or_path` at `google/gemma-4-E2B-it`.
- Targets: the 16-layer majority group exposes **`q_proj, o_proj, gate_proj, up_proj, down_proj`** only — the probe confirmed these layers have **no `k_proj`/`v_proj`** (unified/shared KV), so drop k/v. Start MLP-only (`["down_proj", "gate_proj", "up_proj"]`) to shrink the surface; Phase 3 filtering is still required (probe saw `gate_proj` 6144 vs 12288 across profiles). k/v exist only in the other, smaller profiles.
- Context encoder: early-exit, `layer_idx = text_config.num_hidden_layers // 4`.

### 7.2 Launch script
- **New file**: `scripts/main_exp/gemma4/train_gemma4_e2b.sh`.

---

## Critical Files Summary

| File | Changes | Priority |
|---|---|---|
| `pyproject.toml` / `uv.lock` | transformers `4.51.3` → 5.x; verify `peft` compat | **Blocker** |
| `src/ctx_to_lora/model_loading.py` | register Gemma 4, `Gemma4ForConditionalGeneration` + `.language_model`, skip PEFT wrap, thread `peft_config` | **Critical** |
| `src/ctx_to_lora/modeling/hypernet.py` | non-PEFT branches in `get_init_peft_weights`, `get_hypernet_config` (+`:110` guard), `ModulatedPretrainedModel.__init__`, **and `patch_lora_forward`/`reset`** | **Critical** |
| `src/ctx_to_lora/utils.py` | non-PEFT branches in `get_peft_in_out_features` (use `weight.shape`), `get_peft_modules`, `get_lora_module_names` (`max(idx)+1`) | **Critical** |
| layer filtering (new helper + checkpoint) | profile-group filtering; persist non-contiguous `layer_indices` | **Critical** |
| `run_eval.py` / `src/ctx_to_lora/eval_utils.py` | tower-exclusion regex; reload `layer_indices` | High |
| `train.py` | composite config (`:206`,`:262`,`:278`); confirm `:358-362` non-PEFT branch | High |
| `src/ctx_to_lora/data/definitions.py` | CTX_AFFIXES for Gemma 4 (split tokens, empirical IDs) | High |
| `src/ctx_to_lora/modeling/lora_layer.py` | add `else` to module-name map (`:96-100`); verify `nn.Linear.forward` binding (`:31`,`:55`) | Medium |
| `configs/...gemma4_e2b_closed_qa.yaml`, `scripts/...train_gemma4_e2b.sh` | new config + launch | Medium |

## Risks

1. **Runtime injection coupling** — `patch_lora_forward`/`reset` depend on PEFT `BaseTunerLayer`; the whole attach-point model must be reworked, not just init. *(New top risk.)*
2. **Version bump regressions** — `4.x → 5.x` may break existing Gemma-2/3 + PEFT paths; gate behind a regression run (0.2).
3. **`Gemma4ClippableLinear` vs `nn.Linear`** — false `.in_features`/`.out_features` (use `weight.shape`); and `nn.Linear.forward(self,x)` is invalid if a target isn't an `nn.Linear` subclass. Verify the *decoder's* module class.
4. **Non-contiguous `layer_indices`** — must be identical and persisted across train / checkpoint / eval; mis-mapping is silent.
5. **Eval-side tower contamination** — PEFT at eval matches towers; needs the `model\.`-anchored regex.
6. **Non-PEFT path breadth** — audit every `PeftModel` / `PeftConfig` / `BaseTunerLayer` / `get_peft_model_state_dict` reference.
7. **Two Gemma 4 instances** — base + encoder roughly double memory (~5.1B params each). The encoder is early-exit, so truncating it to the exit depth is a valid mitigation.
8. **PLE on the encoder** — per-layer embeddings shape the activations the hypernetwork consumes; confirm the extraction taps the intended representation.

## Verification

1. **Deps**: `from transformers import Gemma4ForConditionalGeneration` resolves; an existing Gemma-2 config trains + evals unchanged (regression gate).
2. **Load**: `Gemma4ForConditionalGeneration.from_pretrained("google/gemma-4-E2B-it")` works and `.language_model` extracts cleanly (**not** `AutoModelForCausalLM` — see 1.2).
3. **PEFT bypass**: model loads without PEFT; dims read via `weight.shape`; `layer_indices` filtered to the largest profile group.
4. **Hypernet init**: `ModulatedPretrainedModel` constructs with a non-PEFT model + injected `peft_config`; generates valid LoRA weights.
5. **Forward + injection**: `patch_lora_forward` attaches to the bare `nn.Linear` targets; a forward pass produces valid loss.
6. **Training**: loss decreases over initial steps.
7. **Eval**: checkpoint loads with persisted `layer_indices`; PEFT excludes towers; generation produces coherent output from the modulated model.
8. **Context encoder**: Gemma 4 loads as the ctx encoder via `.language_model`, early-exits at `text_config.num_hidden_layers // 4`, and emits a `.last_hidden_state` the hypernetwork consumes; the shared tokenizer handles both roles.
