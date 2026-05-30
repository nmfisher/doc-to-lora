"""Minimal Gemma 4 D2L wiring smoke test.

Loads Gemma 4 as base + encoder, constructs ModulatedPretrainedModel via the
same path train.py takes, runs ONE forward pass with synthetic input + context.
Catches Phase 1-4 wiring bugs (load extraction, non-PEFT helpers, layer
grouping, encoder load) without needing self_gen data.

Run from repo root:
    python scripts/smoke_gemma4.py                            # cuda + bf16 (~20 GB VRAM)
    DEVICE=cpu DTYPE=bfloat16 python scripts/smoke_gemma4.py  # ~20 GB RAM, slow (~10 min)

Env vars:
    MODEL_DIR    base + encoder checkpoint (default google/gemma-4-E2B-it)
    DEVICE       cuda (default) or cpu
    DTYPE        bfloat16 (default), float16, float32

Memory: ~20 GB total (base + encoder copies) at bf16. Use a 24 GB+ GPU.
"""
import logging
import os
import sys

import torch
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()

MODEL = os.environ.get("MODEL_DIR", "google/gemma-4-E2B-it")
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}[os.environ.get("DTYPE", "bfloat16")]
TARGET_MODULES = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


section(f"Step 1: load base + tokenizer ({MODEL}, {DEVICE}, {DTYPE})")
peft_config = get_lora_config(
    MODEL, lora_r=8, lora_dropout=0.0, target_modules=TARGET_MODULES
)
base_model, tokenizer = get_model_and_tokenizer(
    model_name_or_path=MODEL,
    train=True,
    requires_grad=False,
    use_flash_attn=False,
    peft_config=peft_config,
    device=DEVICE,
    dtype=DTYPE,
)
print(f"  base class:  {type(base_model).__name__}")
print(f"  num layers:  {len(base_model.layers)}")
print(f"  hidden_size: {base_model.config.hidden_size}")
print(f"  attn impl:   {base_model.config._attn_implementation}")

section("Step 2: build args + hypernet config (Phase 2b + Phase 3)")
ctx_encoder_args = CtxEncoderArguments(
    ctx_encoder_model_name_or_path=MODEL,
    ctx_encoder_type=CTX_ENCODER_TYPE.EARLY_EXIT,
    layer_idx=8,
)
ctx_cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
if hasattr(ctx_cfg, "text_config"):
    ctx_cfg = ctx_cfg.text_config

hypernet_config = get_hypernet_config(
    base_model,
    ctx_cfg,
    HypernetArguments(),
    AggregatorArguments(),
    ctx_encoder_args,
    peft_config=peft_config,
)
print(f"  layer_indices (len={len(hypernet_config.layer_indices)}): {hypernet_config.layer_indices}")
print(f"  feature_sizes (in, out): {hypernet_config.feature_sizes}")

section("Step 3: construct ModulatedPretrainedModel (loads encoder copy)")
model = ModulatedPretrainedModel(
    base_model,
    hypernet_config,
    ctx_encoder_args,
    use_sequence_packing=False,
)
print(f"  base_model:  {type(model.base_model).__name__}")
print(f"  ctx_encoder: {type(model.ctx_encoder).__name__} -> {type(model.ctx_encoder.base_model).__name__}")
print(f"  hypernet.n_layers: {model.hypernet.n_layers}")
print(f"  hypernet.target_modules: {model.hypernet.target_modules}")

section("Step 4: tiny forward pass")
device = model.device
qa_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(device)
ctx_ids = tokenizer("Paris is a city in France.", return_tensors="pt").input_ids.to(device)
ctx_attn = torch.ones_like(ctx_ids)
ctx_pos = torch.arange(ctx_ids.shape[-1], device=device).unsqueeze(0)

with torch.no_grad():
    out = model(
        ctx_ids=ctx_ids,
        ctx_attn_mask=ctx_attn,
        ctx_position_ids=ctx_pos,
        input_ids=qa_ids,
        attention_mask=torch.ones_like(qa_ids),
        position_ids=torch.arange(qa_ids.shape[-1], device=device).unsqueeze(0),
    )
print(f"  forward returned: {type(out).__name__}")
print(f"  logits shape:     {out.logits.shape}")

section("SMOKE OK")
print("Phases 1-4 wiring verified. Next: generate self_gen data and run a few")
print("training steps via scripts/main_exp/gemma4/train_gemma4_e2b.sh.")
