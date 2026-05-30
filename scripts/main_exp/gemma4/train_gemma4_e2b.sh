#!/bin/bash
#
# Gemma 4 E2B-it doc-to-LoRA training. Mirrors scripts/main_exp/1-train.sh
# with the Gemma 4-specific overrides:
#   - model_name_or_path + ctx_encoder_model_name_or_path explicitly set to
#     the same Gemma 4 checkpoint (avoids the train.py:175 fallback path).
#   - target_modules excludes k/v (see config comment for the dim-profile reason).
#   - quantize_ctx_encoder=False — bitsandbytes 4-bit on Gemma 4 is unverified;
#     flip on once you've smoke-tested it.
#
# Memory: at bf16, base + encoder copies are ~10GB each (5.1B params). 8x
# A100-40GB is the original 1-train.sh assumption; tune --num_processes for
# your hardware.
#
# Prereq: self_gen data under data/raw_datasets/self_gen/google/gemma-4-E2B-it/*
# must exist. See scripts/main_exp/gen_data.sh for the generation pipeline
# (Gemma 4 replaces Gemma 2 as the q-generation model).

port=29051

uv run accelerate launch --config_file accelerate_config.yaml --main_process_port $port \
--num_processes=8 --gpu_ids all train.py \
configs/main_exp/gemma4_e2b_closed_qa.yaml \
--model_name_or_path=google/gemma-4-E2B-it \
--ctx_encoder_model_name_or_path=google/gemma-4-E2B-it \
--target_modules=q_proj,o_proj,gate_proj,up_proj,down_proj --lora_r=8 \
--layer_idx=8 \
--eval_strategy=no --max_qas_len=2048 --max_qas_per_sample=1 \
--per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 \
--max_steps=80000 --gradient_accumulation_steps=8 --max_packed_inp_len=4096 \
--max_packed_ctx_len=4096 --use_per_ctx_average_loss=True --use_kl_loss=True \
--quantize_ctx_encoder=False
