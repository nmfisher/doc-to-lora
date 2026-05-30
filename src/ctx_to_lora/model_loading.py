import logging
import os

import torch
from peft import PeftModel
from peft import get_peft_config as _get_peft_config
from peft.utils import PeftType
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

try:
    from transformers import Gemma4ForConditionalGeneration
except ImportError:  # transformers < 5.5; Phase 0 bump required to use Gemma 4
    Gemma4ForConditionalGeneration = None

logger = logging.getLogger()

GEMMA_VISION_MODELS = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]


def check_is_gemma4(model_name):
    return "gemma-4" in model_name.lower()


def check_is_vision_model(model_name):
    return model_name in GEMMA_VISION_MODELS or check_is_gemma4(model_name)


def get_model_and_tokenizer(
    model_name_or_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    tokenizer_kwargs=None,
    use_q_lora=False,
    device="cuda",
    dtype=torch.bfloat16,
):
    model = get_model(
        model_name_or_path,
        train,
        requires_grad,
        use_flash_attn,
        peft_config,
        model_kwargs,
        use_q_lora,
        device,
        dtype,
    )
    tokenizer = get_tokenizer(model_name_or_path, tokenizer_kwargs, peft_config, train)
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    # Gemma 4: also propagate pad_token_id onto the full wrapper, since
    # generate() flows through it (see get_model for the delegation setup).
    wrapper = getattr(model, "_gemma4_full_wrapper", None)
    if wrapper is not None:
        wrapper.config.pad_token_id = tokenizer.pad_token_id
        if getattr(wrapper, "generation_config", None):
            wrapper.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def get_tokenizer(
    model_name_or_path, tokenizer_kwargs=None, peft_config=None, train=False
):
    padding_side = "left" if not train else "right"
    truncation_side = "left"

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        add_bos_tokens=False,
        add_eos_tokens=False,
        padding_side=padding_side,
        truncation_side=truncation_side,
        trust_remote_code=True,
        **tokenizer_kwargs,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    template_path = f"chat_templates/{model_name_or_path}.jinja"
    if not os.path.exists(template_path):
        logger.warning(
            f"Chat template not found at {template_path}. Using default template."
        )
        return tokenizer

    logger.info(f"Using chat template from {template_path}")
    chat_template = open(template_path).read()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    tokenizer.chat_template = chat_template
    return tokenizer


def get_model(
    model_name_or_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    use_q_lora=False,
    device="cuda",
    dtype=torch.bfloat16,
):
    model_init_kwargs = dict(
        pretrained_model_name_or_path=model_name_or_path,
        device_map=device,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    is_vision_model = check_is_vision_model(model_name_or_path)
    if model_kwargs is not None:
        model_init_kwargs.update(model_kwargs)

    is_bidir_model = (
        "bert" in model_name_or_path.lower() or "gte" in model_name_or_path.lower()
    )

    if use_flash_attn:
        if check_is_gemma4(model_name_or_path):
            # Gemma 4 ships its own optimized attention; keep "eager"
            pass
        elif "gte" not in model_name_or_path:
            model_init_kwargs["attn_implementation"] = "flash_attention_2"
        elif "gte" in model_name_or_path:
            model_init_kwargs["attn_implementation"] = "sdpa"

    if is_bidir_model:
        # bidir encoders (BERT/GTE) want fp32
        model_init_kwargs["dtype"] = torch.float32

    if use_q_lora:
        # https://huggingface.co/blog/4bit-transformers-bitsandbytes
        # https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
        # see bitsandbytes for the quantization implementation https://github.com/bitsandbytes-foundation/bitsandbytes
        # see unsloth https://huggingface.co/docs/trl/v0.7.11/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth
        # does work currently bc it modifies the forward pass call of Linear
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_init_kwargs["quantization_config"] = bnb_config

    logger.debug(f"Model init kwargs: {model_init_kwargs}")
    is_gemma4 = check_is_gemma4(model_name_or_path)
    if not is_vision_model:
        if is_bidir_model:
            model = AutoModel.from_pretrained(**model_init_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(**model_init_kwargs)
    elif is_gemma4:
        if Gemma4ForConditionalGeneration is None:
            raise ImportError(
                "Gemma 4 requires transformers>=5.5 (do the Phase 0 bump)."
            )
        full_model = Gemma4ForConditionalGeneration.from_pretrained(**model_init_kwargs)
        # Gemma 4 nests one level deeper than Gemma 3 (probe-confirmed):
        # full_model.model.language_model -> Gemma4TextModel
        model = full_model.model.language_model
        # Gemma4TextModel doesn't subclass GenerationMixin, so .generate is
        # missing — would crash eval at hypernet.py's `self.base_model.generate(...)`.
        # Delegate to the full wrapper (which inherits GenerationMixin and routes
        # its forward back through this same text model, picking up our LoRA
        # patches). Bare + wrapper share weights, so this only keeps a Python
        # object alive — no extra GPU memory.
        model._gemma4_full_wrapper = full_model
        model.generate = full_model.generate
    else:
        model = Gemma3ForConditionalGeneration.from_pretrained(**model_init_kwargs)
        model = model.language_model
    # Skip PEFT wrap for Gemma 4: the hypernet injects LoRA at runtime via the
    # non-PEFT path (Phase 2). peft_config is still threaded by the caller for
    # the hypernet to consume directly.
    if peft_config is not None and not is_gemma4:
        model = PeftModel(model, peft_config)
    model.train(train)
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad
    return model


def get_lora_config(model_dir, **kwargs):
    if "target_modules" not in kwargs or kwargs["target_modules"] is None:
        logger.info("No target modules specified for LoRA.")
        return None
    r = kwargs.pop("lora_r", 8)
    peft_conf_kwargs = dict(
        r=r,
        peft_type=PeftType.LORA,
        base_model_name_or_path=model_dir,
        task_type="CAUSAL_LM",
        lora_dropout=kwargs.get("lora_dropout", 0.0),
        lora_alpha=r ** (3 / 2) * 2,
    )

    peft_conf_kwargs.update(kwargs)
    peft_config = _get_peft_config(peft_conf_kwargs)
    return peft_config
