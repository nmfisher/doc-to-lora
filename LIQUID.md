# Plan: Train a Doc-to-LoRA Hypernetwork for LFM2.5-8B-A1B

## Context

The user wants to train a hypernetwork (based on the Doc-to-LoRA approach from Sakana AI) for Liquid AI's LFM2.5-8B-A1B model. The Doc-to-LoRA codebase at `/Volumes/T7/projects/doc-to-lora` currently supports standard transformer models (Gemma, Mistral, Qwen). LFM2.5 uses a fundamentally different hybrid architecture (MoE + LIV convolution + GQA) that requires significant adaptation. Training will use the full-precision model `LiquidAI/LFM2.5-8B-A1B` (not the GGUF variant, which is inference-only) on 8x GPU hardware, with optional GGUF export for deployment.

**Key challenge**: LFM2.5 has 24 layers but only 6 use attention — the other 18 use double-gated LIV convolution blocks. Module naming, LoRA targeting, and context encoding all differ from standard transformers.

---

## Phase 1: Environment Setup & Model Investigation

### 1.1 Upgrade `transformers` to v5.x
- **File**: `pyproject.toml` — change `transformers==4.51.3` to `transformers>=5.9.0`
- LFM2.5's model class `Lfm2MoeForCausalLM` only exists in transformers v5+
- This is the most consequential dependency change; verify no API breakage in the codebase's usage of `Trainer`, `AutoModelForCausalLM`, `AutoTokenizer`, etc.

### 1.2 Verify PEFT compatibility with `lfm2_moe`
- Check if current PEFT version supports the `lfm2_moe` model type for LoRA application
- If not, may need to upgrade PEFT or manually register the model type
- Critical check: PEFT must be able to wrap `Lfm2MoeForCausalLM` and inject LoRA into its `nn.Linear` submodules

### 1.3 Inspect model architecture empirically
- **Write a small script** to load the model and print its module tree, identifying:
  - Exact module names and paths for all `nn.Linear` layers
  - Which layers are conv vs attention (the `config.layer_types` field)
  - Expert parameter structure (confirmed: MoE expert weights are `nn.Parameter`, not `nn.Linear` — cannot be LoRA targets)
  - The model's `hidden_size` (expected: 2048), `num_hidden_layers` (24), etc.
- This confirms the module mapping before any code changes

**Expected module structure per layer type:**

| Layer type (18 conv) | Module path | Type |
|---|---|---|
| Conv input projection | `model.layers[i].conv.in_proj` | `nn.Linear` ✓ |
| Conv output projection | `model.layers[i].conv.out_proj` | `nn.Linear` ✓ |
| MoE gate | `model.layers[i].feed_forward.gate` | `nn.Linear` ✓ |
| MoE experts | `feed_forward.experts.gate_up_proj` | `nn.Parameter` ✗ |
| MoE experts | `feed_forward.experts.down_proj` | `nn.Parameter` ✗ |

| Layer type (6 attention, at indices 2,6,10,14,19,22) | Module path | Type |
|---|---|---|
| Q projection | `model.layers[i].self_attn.q_proj` | `nn.Linear` ✓ |
| K projection | `model.layers[i].self_attn.k_proj` | `nn.Linear` ✓ |
| V projection | `model.layers[i].self_attn.v_proj` | `nn.Linear` ✓ |
| Output projection | `model.layers[i].self_attn.out_proj` | `nn.Linear` ✓ |

**Recommended LoRA targets**: `out_proj` (present in all 24 layers — both conv and attention). Alternative: `["out_proj", "in_proj"]` for broader coverage.

---

## Phase 2: Core Codebase Adaptation

### 2.1 Generalize module path resolution in `lora_layer.py`
- **File**: `src/ctx_to_lora/modeling/lora_layer.py` (lines 96–100)
- **Problem**: Hardcoded mapping assumes `self_attn.{name}` or `mlp.{name}` — doesn't handle `conv.in_proj`, `conv.out_proj`, `feed_forward.w1`, etc.
- **Fix**: Replace the if/elif block with a dynamic resolver function:
  ```python
  def resolve_module_path(layer, module_name):
      candidates = [
          f"self_attn.{module_name}",     # standard attention + LFM2 attention
          f"conv.{module_name}",          # LFM2 conv block
          f"mlp.{module_name}",           # standard MLP
          f"feed_forward.{module_name}",  # LFM2 dense MLP
      ]
      for candidate in candidates:
          try:
              return candidate, attrgetter(candidate)(layer)
          except AttributeError:
              continue
      raise AttributeError(f"Cannot resolve '{module_name}' in {type(layer).__name__}")
  ```
- Apply this resolver in `apply_lora_to_layers()` wherever `long_mname` is constructed

### 2.2 Verify `utils.py` compatibility
- **File**: `src/ctx_to_lora/utils.py`
- `get_layers()` (line 42): recurses through `model.model` to find `.layers` — should work since `Lfm2MoeForCausalLM` has `model.model.layers`
- `get_peft_in_out_features()` (line 174): asserts `isinstance(module.base_layer, torch.nn.Linear)` — works for `in_proj`, `out_proj`, `q_proj`, etc.
- `get_peft_modules()` (line 203): uses `get_layers()` and iterates modules — verify it correctly finds `conv.out_proj`, `self_attn.q_proj`, etc.

### 2.3 Handle context encoder output type
- **File**: `src/ctx_to_lora/modeling/ctx_encoder.py`
- `EarlyExit` (line 59): truncates `base_model.layers` to first L//4 and calls forward
- LFM2 MoE returns `MoeModelOutputWithPast` (not `BaseModelOutputWithPast`) but it has `.last_hidden_state`, so the existing `.last_hidden_state` access should work
- First 6 layers (L//4 of 24) are: conv, conv, attention, conv, conv, conv — the mixed types are handled internally by `Lfm2MoeDecoderLayer.forward()`
- **Action**: Add an assertion or check that the output type has `last_hidden_state`; handle `MoeModelOutputWithPast` if needed

### 2.4 Update `model_loading.py`
- **File**: `src/ctx_to_lora/model_loading.py`
- The vision model check (`is_vision_model`) may not recognize `lfm2_moe` — add it to the exclusion list
- Flash attention: `Lfm2MoeAttention` supports flash attention; conv layers ignore the `attn_implementation` kwarg, so passing `flash_attention_2` is safe
- Ensure `trust_remote_code=True` is passed (already the case in the codebase)

### 2.5 Handle `num_hidden_layers` and hidden size
- `train.py` line 190: `ctx_encoder_args.layer_idx = config.num_hidden_layers // 4` → 24 // 4 = 6 ✓
- `aggregator.py` line 66: reads `config.hidden_size` → 2048 ✓
- `train.py` line 262: reads `config.max_position_embeddings` → 128000 (much larger than default `max_packed_ctx_len` of 32768, so fine) ✓

---

## Phase 3: Data Pipeline

### 3.1 Add LFM2.5 chat template
- **New file**: `chat_templates/LiquidAI/LFM2.5-8B-A1B.jinja`
- Uses ChatML format with special tokens: `<|startoftext|>`, `<|im_start|>`, `<|im_end|>`
- Must support `{% generation %}` blocks for the existing template engine
- LFM2.5 is a reasoning model — the template should handle `<think_reasoning>...</think_reasoning>` tags (optionally strip reasoning for training)

### 3.2 Add context chunk affixes
- **File**: `src/ctx_to_lora/data/definitions.py`
- Add a `CTX_AFFIXES` entry for `LiquidAI/LFM2.5-8B-A1B` with the correct token IDs for:
  - Prefix: `<|startoftext|><|im_start|>system\n...\n<|im_end|>\n`
  - Suffix: `<|im_end|>\n<|im_start|>user\n`
- Token IDs must be computed empirically by running the tokenizer on these fragments

### 3.3 Update self-generation data pipeline
- **File**: `data/self_generate_qa.py`
- Ensure vLLM supports LFM2.5 for fast inference during data generation
- Handle reasoning output: set `preserve_thinking=false` or strip reasoning traces from generated QA pairs
- LFM2.5 recommended generation params: `temperature=0.2`, `top_k=80`, `repetition_penalty=1.05`

### 3.4 Reasoning model handling
- Since LFM2.5-8B-A1B is a reasoning model, responses include chain-of-thought before the final answer
- For training: strip reasoning traces from labels (or include them, depending on objectives)
- For evaluation: strip reasoning from generated text before computing metrics
- Alternative: use `LiquidAI/LFM2.5-8B-A1B-Base` (non-reasoning) if available for simpler training

---

## Phase 4: Training Configuration

### 4.1 Create training config
- **New file**: `configs/main_exp/lfm25_8b_a1b_closed_qa.yaml`

```yaml
# Model
model_name_or_path: LiquidAI/LFM2.5-8B-A1B
ctx_model_name_or_path: LiquidAI/LFM2.5-8B-A1B  # same model for context encoder

# LoRA
lora_r: 8
lora_dropout: 0.0
target_modules:
  - out_proj              # present in all 24 layers

# Experiment
exp_setup: hyperlora
per_rank_gen: true
per_layer_processing: true
gen_lora_l1_reg_coef: 0.1

# Context encoder
ctx_encoder_type: early_exit    # layer_idx auto-set to 6

# Aggregator
n_latent_queries: 8             # matches 24 layers' worth of modules
num_blocks: 9
latent_size: 512

# Training
bf16: true
learning_rate: 4.0e-5
optim: adamw_torch_fused
lr_scheduler_type: cosine_with_min_lr
warmup_steps: 100
max_steps: 80000
neftune_noise_alpha: 5.0

# Sequences
use_sequence_packing: true
max_packed_inp_len: 6144
max_packed_ctx_len: 6144
gradient_accumulation_steps: 11
per_device_train_batch_size: 1

# Loss
use_kl_loss: true

# Data
train_ds_names: [...]   # self-gen QA data for LFM2.5
val_ds_names:
  - squad
  - drop
  - ropes
```

### 4.2 Key hyperparameter notes
- **LoRA rank 8**: With `hidden_size=2048` and MoE active params ~1.5B, rank 8 provides good capacity
- **Target `out_proj`**: Uniform across all 24 layers (both conv and attention types) — analogous to current default of `down_proj`
- **Learning rate 4e-5**: Same as existing Gemma-2-2B setup; the MoE model's smaller active parameter count means the effective model being adapted is smaller
- **L1 reg 0.1**: Same as main experiment to keep generated LoRA weights sparse

---

## Phase 5: Training Script & Launch

### 5.1 Create training launch script
- **New file**: `scripts/main_exp/train_lfm25_8b_a1b.sh`
- Uses `accelerate launch` with 8 GPUs, matching existing pattern from `scripts/main_exp/1-train.sh`

```bash
accelerate launch train.py configs/main_exp/lfm25_8b_a1b_closed_qa.yaml \
  --model_name_or_path=LiquidAI/LFM2.5-8B-A1B \
  --per_rank_gen=True \
  --per_layer_processing=True \
  --gen_lora_l1_reg_coef=0.1 \
  --max_steps=80000
```

### 5.2 Data generation
- Either download pre-generated data (if available for LFM2.5) or generate using `data/self_generate_qa.py` with vLLM
- Expected ~100GB of self-gen QA data (matching the scale for other models)

---

## Phase 6: Evaluation & Deployment

### 6.1 Evaluation pipeline
- **File**: `src/ctx_to_lora/eval_utils.py`
- Update generation decoding to handle LFM2.5's token format and strip reasoning traces
- Create evaluation script: `scripts/main_exp/eval/d2l_lfm25.sh`

### 6.2 GGUF deployment path (optional)
1. Train hypernetwork on `LiquidAI/LFM2.5-8B-A1B` (safetensors)
2. At inference, hypernetwork generates LoRA A/B matrices from context
3. Export LoRA weights and apply to GGUF model via llama.cpp's `--lora` flag
4. Module name mapping may be needed between safetensors and GGUF naming conventions

---

## Critical Files Summary

| File | Action | Priority |
|---|---|---|
| `pyproject.toml` | Upgrade transformers to >=5.9.0 | **Critical** |
| `src/ctx_to_lora/modeling/lora_layer.py` | Replace hardcoded module path resolution | **Critical** |
| `src/ctx_to_lora/utils.py` | Verify get_layers/get_peft_modules work with lfm2_moe | High |
| `src/ctx_to_lora/model_loading.py` | Handle lfm2_moe model type, flash attn | High |
| `src/ctx_to_lora/data/definitions.py` | Add CTX_AFFIXES for LFM2.5 | High |
| `chat_templates/LiquidAI/LFM2.5-8B-A1B.jinja` | Create new chat template | High |
| `src/ctx_to_lora/modeling/ctx_encoder.py` | Verify MoeModelOutputWithPast compatibility | Medium |
| `src/ctx_to_lora/eval_utils.py` | Handle reasoning output format | Medium |
| `configs/main_exp/lfm25_8b_a1b_closed_qa.yaml` | New training config | Medium |
| `scripts/main_exp/train_lfm25_8b_a1b.sh` | New launch script | Medium |
| `data/self_generate_qa.py` | Update for LFM2.5 template/reasoning | Medium |

## Risks

1. **transformers v5 API changes** — highest risk; may break existing code that uses internal transformers APIs. Mitigate by testing incrementally.
2. **PEFT lacking lfm2_moe support** — if PEFT can't apply LoRA to this model type, need to manually register or inject LoRA layers.
3. **Conv layer behavior during early-exit context encoding** — the truncated model must produce valid hidden states. Verify empirically.
4. **Reasoning model complexity** — LFM2.5's chain-of-thought adds complexity to data generation and evaluation. Consider using a non-reasoning variant if available.

## Verification

1. **Phase 1 checkpoint**: Model loads with transformers v5, PEFT wraps it, and `model.generate()` produces coherent text
2. **Phase 2 checkpoint**: LoRA layers inject correctly into conv and attention modules; forward pass produces valid loss values
3. **Phase 3 checkpoint**: Chat template produces correct token sequences; context chunking preserves meaning
4. **Phase 4-5 checkpoint**: Training runs without errors, loss decreases, W&B logs look reasonable
5. **Phase 6 checkpoint**: Evaluation metrics (per-token accuracy, prefix matching) on SQuAD/DROP are above base model baseline
