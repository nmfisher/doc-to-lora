"""Probe a raw Gemma 4 checkpoint to validate the assumptions in GEMMA4_SUPPORT.md.

Read-only: loads the model once and reports the empirical facts the plan depends on,
WITHOUT touching any not-yet-written non-PEFT code. Run this before implementing.

What it answers:
  - Does the model download/load, with which transformers version and class?
  - Is `Gemma4ForConditionalGeneration` importable? Does `.language_model` extract?
  - Real layer count (plan assumes 35; transformers config default was 30).
  - Per-layer dimension profiles -> how many groups, how big is the largest (plan: ~4 / 16-of-35)?
  - Are decoder target modules `nn.Linear` or `Gemma4ClippableLinear`?
  - Do `.in_features`/`.out_features` disagree with `weight.shape` (the b7051e5 lesson)?
  - Real control-token IDs (split form `<|turn>`...`<turn|>`, not `<|turn|>`).
  - Early-exit hidden state: is the residual width uniform across layers?

Usage:
  uv run python scripts/probe_gemma4.py                      # CPU fp32 (0 VRAM, ~22GB RAM)
  DEVICE=cuda DTYPE=bfloat16 uv run python scripts/probe_gemma4.py   # ~12GB VRAM
  DEVICE=meta uv run python scripts/probe_gemma4.py          # ~0 memory: structural checks
                                                             # only (skips the forward pass)
"""

import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

MODEL_DIR = os.environ.get("MODEL_DIR", "google/gemma-4-E2B-it")
DEVICE = os.environ.get("DEVICE", "cpu")
META = DEVICE == "meta"  # build architecture only (no real weights) -> ~0 memory
DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
    os.environ.get("DTYPE", "float32")
]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        logger.info(f"  PASS: {name}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        logger.error(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))
    return condition


def section(title):
    logger.info(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
section("Step 0: Environment")
# ---------------------------------------------------------------------------
import transformers

logger.info(f"  transformers: {transformers.__version__}")
logger.info(f"  torch:        {torch.__version__}")
logger.info(f"  model:        {MODEL_DIR}   device={DEVICE} dtype={DTYPE}")

try:
    from transformers import Gemma4ForConditionalGeneration  # noqa: F401

    has_gemma4_class = True
except Exception as e:
    has_gemma4_class = False
    logger.info(f"  (Gemma4ForConditionalGeneration import failed: {e})")
ok_g4 = check("Gemma4ForConditionalGeneration importable", has_gemma4_class)
if not ok_g4:
    logger.error("    transformers lacks the gemma4 module -> do the Phase 0 bump (>=5.5)")

# Is Gemma 4 registered as a vision/multimodal model in this repo yet? (expected: not yet)
try:
    from ctx_to_lora.model_loading import GEMMA_VISION_MODELS, check_is_vision_model

    logger.info(f"  repo GEMMA_VISION_MODELS: {GEMMA_VISION_MODELS}")
    logger.info(f"  check_is_vision_model({MODEL_DIR}) = {check_is_vision_model(MODEL_DIR)}  "
                f"(expected False until Phase 1.1)")
except Exception as e:
    logger.info(f"  (could not import repo model_loading: {e})")

# ---------------------------------------------------------------------------
section("Step 1: Load full (multimodal) model")
# ---------------------------------------------------------------------------
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

full_model = None
try:
    if META:
        cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
        with torch.device("meta"):
            if has_gemma4_class:
                full_model = Gemma4ForConditionalGeneration(cfg)
                load_path = "Gemma4ForConditionalGeneration(config) on meta"
            else:
                full_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
                load_path = "AutoModelForCausalLM.from_config on meta"
        full_model = full_model.eval()
    else:
        load_kwargs = dict(
            torch_dtype=DTYPE, low_cpu_mem_usage=True, trust_remote_code=True,
            output_hidden_states=True, output_attentions=False,
        )
        if has_gemma4_class:
            full_model = Gemma4ForConditionalGeneration.from_pretrained(MODEL_DIR, **load_kwargs)
            load_path = "Gemma4ForConditionalGeneration"
        else:
            full_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **load_kwargs)
            load_path = "AutoModelForCausalLM"
        full_model = full_model.to(DEVICE).eval()
    check("Model loaded", True, f"via {load_path}, type={type(full_model).__name__}")
except Exception as e:
    check("Model loaded", False, str(e))
    logger.error("  Cannot continue without a model (check HF access / transformers version).")
    if META:
        logger.error("  (meta build can hit non-meta-safe init on some versions — try DEVICE=cpu instead.)")
    sys.exit(1)

check("NOT PEFT-wrapped", not hasattr(full_model, "peft_config"))
logger.info(f"  config type: {type(full_model.config).__name__}")

# Multimodal towers? (relevant to eval tower-exclusion + the multimodal-encoder opt-in)
for attr in ["vision_tower", "audio_tower", "multi_modal_projector"]:
    present = hasattr(full_model, attr) or hasattr(getattr(full_model, "model", object()), attr)
    logger.info(f"  has {attr}: {present}")

# ---------------------------------------------------------------------------
section("Step 2: Composite config")
# ---------------------------------------------------------------------------
cfg = full_model.config
has_text_config = hasattr(cfg, "text_config")
check("config has .text_config (composite)", has_text_config)
text_cfg = cfg.text_config if has_text_config else cfg

top_nhl = getattr(cfg, "num_hidden_layers", None)
text_nhl = getattr(text_cfg, "num_hidden_layers", None)
logger.info(f"  num_hidden_layers: top-level={top_nhl}  text_config={text_nhl}")
logger.info(f"  hidden_size:       {getattr(text_cfg, 'hidden_size', None)}")
logger.info(f"  head_dim:          {getattr(text_cfg, 'head_dim', None)}  "
            f"global_head_dim={getattr(text_cfg, 'global_head_dim', None)}")
logger.info(f"  sliding_window:    {getattr(text_cfg, 'sliding_window', None)}  "
            f"pattern={getattr(text_cfg, 'sliding_window_pattern', None)}")
logger.info(f"  max_position_embeddings: {getattr(text_cfg, 'max_position_embeddings', None)}")
logger.info(f"  vocab_size:        {getattr(text_cfg, 'vocab_size', None)}")

# ---------------------------------------------------------------------------
section("Step 3: Extract .language_model + get_layers")
# ---------------------------------------------------------------------------
lm = None
for path in ["language_model", "model.language_model"]:
    obj = full_model
    try:
        for part in path.split("."):
            obj = getattr(obj, part)
        lm = obj
        check(".language_model extracted", True, f"via .{path}, type={type(lm).__name__}")
        break
    except AttributeError:
        continue
if lm is None:
    check(".language_model extracted", False, "no .language_model / .model.language_model")
    lm = full_model  # fall back so later steps still probe something

try:
    from ctx_to_lora.utils import get_layers

    layers = get_layers(lm)
    check("repo get_layers() resolves", True, f"{len(layers)} layers")
except Exception as e:
    check("repo get_layers() resolves", False, str(e))
    # inline fallback so the probe continues
    obj = lm
    while hasattr(obj, "model"):
        obj = obj.model
    layers = obj.layers
    logger.info(f"  (fallback layer finder: {len(layers)} layers)")

n_layers = len(layers)
check("layer count matches text_config.num_hidden_layers", n_layers == text_nhl,
      f"get_layers={n_layers}, config={text_nhl}")

# ---------------------------------------------------------------------------
section("Step 4: Per-layer dimension profiles (heterogeneity)")
# ---------------------------------------------------------------------------
layer_dims = {}  # idx -> {module_short_name: (in, out)} from weight.shape
for i, layer in enumerate(layers):
    dims = {}
    for name, module in layer.named_modules():
        short = name.split(".")[-1]
        if short in TARGET_MODULES and hasattr(module, "weight") and getattr(module, "weight").dim() == 2:
            w = module.weight.shape  # [out, in]
            dims[short] = (w[1], w[0])
    layer_dims[i] = dims

groups = {}
for idx, dims in layer_dims.items():
    groups.setdefault(tuple(sorted(dims.items())), []).append(idx)

logger.info(f"  distinct dimension profiles: {len(groups)}")
for key, idxs in sorted(groups.items(), key=lambda x: -len(x[1])):
    s = dict(key)
    logger.info(f"    {len(idxs):2d} layers {idxs}")
    logger.info(f"        q_proj={s.get('q_proj')} k_proj={s.get('k_proj')} "
                f"v_proj={s.get('v_proj')} gate_proj={s.get('gate_proj')} down_proj={s.get('down_proj')}")

largest_group = max(groups.values(), key=len)
ref_idx = largest_group[0]
ref_dims = layer_dims[ref_idx]
incompatible = sorted(set(range(n_layers)) - set(largest_group))
layer_indices = sorted(largest_group)

check("heterogeneous layers present", len(groups) > 1, f"{len(groups)} profiles")
logger.info(f"  majority group: {len(largest_group)}/{n_layers} layers -> {layer_indices}")
if incompatible:
    logger.info(f"  incompatible (dropped): {incompatible}")
contiguous = layer_indices == list(range(layer_indices[0], layer_indices[-1] + 1))
logger.info(f"  majority-group indices contiguous? {contiguous}  "
            f"(non-contiguous => Phase 3 array-indexing care)")
# Do some layers lack target modules entirely? (=> KeyError risk)
missing = {tm: [i for i in layer_indices if tm not in layer_dims[i]] for tm in TARGET_MODULES}
missing = {tm: v for tm, v in missing.items() if v}
if missing:
    logger.info(f"  modules absent from some kept layers: {missing}")

# ---------------------------------------------------------------------------
section("Step 5: Module class + .in_features vs weight.shape (b7051e5 lesson)")
# ---------------------------------------------------------------------------
ref_layer = layers[ref_idx]
ref_modules = {name.split(".")[-1]: m for name, m in ref_layer.named_modules()
               if name.split(".")[-1] in TARGET_MODULES and hasattr(m, "weight")}
logger.info(f"  reference layer index: {ref_idx}")
any_clippable = False
any_mismatch = False
for tm in TARGET_MODULES:
    m = ref_modules.get(tm)
    if m is None:
        logger.info(f"    {tm}: (absent in ref layer)")
        continue
    cls = type(m).__name__
    is_linear = isinstance(m, torch.nn.Linear)
    any_clippable = any_clippable or ("Clippable" in cls) or (not is_linear)
    w_in, w_out = m.weight.shape[1], m.weight.shape[0]
    a_in = getattr(m, "in_features", None)
    a_out = getattr(m, "out_features", None)
    mismatch = (a_in is not None and a_in != w_in) or (a_out is not None and a_out != w_out)
    any_mismatch = any_mismatch or mismatch
    flag = "  <-- MISMATCH" if mismatch else ""
    logger.info(f"    {tm}: {cls}  nn.Linear={is_linear}  "
                f"weight=({w_in},{w_out}) attrs=({a_in},{a_out}){flag}")

ok_linear = check("decoder targets are nn.Linear (forward-patch via nn.Linear.forward is safe)",
                  not any_clippable)
if not ok_linear:
    logger.error("    non-nn.Linear targets found -> Phase 2.5 must call the module's own forward")
if any_mismatch:
    logger.info("  -> .in_features/.out_features are UNRELIABLE here; use weight.shape (confirms Phase 2.1)")
else:
    logger.info("  -> .in_features/.out_features agree with weight.shape on this model")

# ---------------------------------------------------------------------------
section("Step 6: Control tokens + empirical chat-template affixes")
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, padding_side="right")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

unk = tokenizer.unk_token_id


def exists(tok):
    tid = tokenizer.convert_tokens_to_ids(tok)
    ok = tid is not None and tid != unk
    return ok, tid


logger.info("  split form (plan v2 says THESE are real):")
for t in ["<|turn>", "<turn|>", "<|tool_call>", "<tool_call|>", "<|tool>", "<tool|>"]:
    ok, tid = exists(t)
    logger.info(f"    {t:14s} exists={ok}  id={tid}")
logger.info("  single form (plan v1 guessed these — expected to NOT exist):")
for t in ["<|turn|>", "<|end_turn|>", "<|think|>"]:
    ok, tid = exists(t)
    logger.info(f"    {t:14s} exists={ok}  id={tid}")
logger.info("  gemma 2/3 form (in case retained):")
for t in ["<start_of_turn>", "<end_of_turn>"]:
    ok, tid = exists(t)
    logger.info(f"    {t:16s} exists={ok}  id={tid}")

logger.info("\n  empirical apply_chat_template (read CTX_AFFIXES prefix/suffix off this):")
try:
    chat = [{"role": "user", "content": "PROMPT"}, {"role": "assistant", "content": "REPLY"}]
    ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    try:                               # dict / BatchEncoding (a UserDict, not a dict) -> id list
        ids = ids["input_ids"]
    except (TypeError, KeyError):
        pass                           # already a plain list
    if ids and isinstance(ids[0], (list, tuple)):  # un-batch if nested
        ids = ids[0]
    ids = [int(t) for t in ids]
    logger.info(f"    full user+assistant ids: {ids}")
    for tid in ids:
        logger.info(f"      {tid:>7d}  {tokenizer.decode([tid])!r}")
    check("apply_chat_template works", True)
except Exception as e:
    check("apply_chat_template works", False, str(e))

# ---------------------------------------------------------------------------
section("Step 7: Forward pass + early-exit hidden state")
# ---------------------------------------------------------------------------
if META:
    logger.info("  SKIPPED (meta mode has no real weights to run a forward through).")
    logger.info(f"  This step only verifies the residual width is uniform = hidden_size="
                f"{getattr(text_cfg, 'hidden_size', None)} (true by construction). "
                f"Re-run with DEVICE=cpu/cuda to confirm empirically.")
else:
    try:
        enc = tokenizer("The capital of France is", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = lm(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"),
                     output_hidden_states=True)
        hs = out.hidden_states  # tuple: embeddings + one per layer
        widths = sorted({h.shape[-1] for h in hs})
        check("residual width uniform across all layers", len(widths) == 1,
              f"widths={widths} (Phase 4.3: early-exit extraction is dim-safe iff uniform)")
        early = (text_nhl or n_layers) // 4
        logger.info(f"  hidden_states tuple length: {len(hs)} (= n_layers+1)")
        logger.info(f"  early-exit layer_idx (n//4): {early} -> hidden state shape {tuple(hs[early].shape)}")
        check("output exposes .last_hidden_state", hasattr(out, "last_hidden_state"),
              "Phase 4.2: early-exit ctx encoder reads .last_hidden_state")
    except Exception as e:
        check("forward pass + hidden states", False, str(e))
        import traceback
        traceback.print_exc()

# ---------------------------------------------------------------------------
section("Step 8: Eval-path readiness — .generate() on the extracted model")
# ---------------------------------------------------------------------------
# ModulatedPretrainedModel.generate calls self.base_model.generate(...). For
# Gemma 3 the extraction lands on Gemma3ForCausalLM (inherits GenerationMixin).
# For Gemma 4 we go one level deeper (model.model.language_model); the
# resulting class decides whether eval works as-is or needs a wrapper.
lm_class = type(lm).__name__
has_generate = callable(getattr(lm, "generate", None))
logger.info(f"  bare extracted class: {lm_class}, has .generate: {has_generate}")
if not has_generate:
    logger.info(
        "  -> Bare model has no GenerationMixin. model_loading.py works around"
        " this by delegating .generate from the bare model to the full"
        " Gemma4ForConditionalGeneration wrapper (which shares weights with"
        " the bare model). Verifying that pattern below."
    )

# Mirror the repo's delegation pattern (model_loading.py:get_model Gemma 4 branch).
delegated_ok = False
if full_model is not None and callable(getattr(full_model, "generate", None)):
    lm._gemma4_full_wrapper = full_model
    lm.generate = full_model.generate
    delegated_ok = callable(getattr(lm, "generate", None))
check(
    "generate accessible via wrapper-delegation pattern",
    delegated_ok,
    "model_loading.py uses `model.generate = full_model.generate`",
)

if META:
    logger.info("  SKIPPED real .generate() call (meta mode has no weights).")
elif not delegated_ok:
    logger.info("  SKIPPED real .generate() call (delegation pattern not available).")
else:
    try:
        enc = tokenizer("The capital of France is", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = lm.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                max_new_tokens=4,
                do_sample=False,
            )
        new_tokens = gen[0, enc["input_ids"].shape[-1]:].tolist()
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
        check(
            "tiny generation via delegation succeeded",
            len(new_tokens) > 0,
            f"new_tokens={new_tokens} -> {decoded!r}",
        )
    except Exception as e:
        check("tiny generation via delegation succeeded", False, str(e))
        import traceback
        traceback.print_exc()

# ---------------------------------------------------------------------------
section(f"SUMMARY: {passed} passed, {failed} failed")
# ---------------------------------------------------------------------------
logger.info("Compare the numbers above against GEMMA4_SUPPORT.md assumptions:")
logger.info(f"  - layer count: {n_layers}  (plan assumed 35)")
logger.info(f"  - dimension profiles: {len(groups)}  (plan assumed ~4)")
logger.info(f"  - majority group: {len(largest_group)}/{n_layers}  (plan assumed 16/35)")
logger.info("  - decoder target class + in_features reliability: see Step 5")
logger.info("  - real control tokens: see Step 6")
logger.info("  - eval-time .generate() availability: see Step 8")
sys.exit(1 if failed else 0)
