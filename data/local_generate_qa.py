#!/usr/bin/env python3
"""Local-teacher answer + logprob generation for distillation training.

Counterpart to data/self_generate_qa.py, which uses vLLM (Volta+ only).
This script talks to a llama.cpp / Ollama HTTP server over the OpenAI-
compatible chat-completions endpoint, so it runs on Pascal GPUs (GTX
1080 Ti etc.) via GGUF / Q4 quantization where vLLM can't.

Pipeline:
  1. Load a HF parquet dataset with (context, prompts, responses) columns
     (e.g., cocoon_code_qa).
  2. For every (context, question) pair, POST to the server with
     `logprobs=true, top_logprobs=K` to get the teacher's answer
     AND the top-K logprobs per generated token.
  3. Re-encode the token strings back to vocab IDs via the local HF
     tokenizer. The vocab MUST match the downstream student model, or
     KL distillation is computing divergence over mismatched spaces.
  4. Write a parquet matching self_generate_qa.py's output schema:
     (context, ctx_ids, input_ids, response_start_end,
      logprobs_vals, logprobs_indices), which packing.py's has_logprobs
     branch already knows how to consume, and DistillationTrainer's
     KL-loss path already knows how to use.

Server prerequisites — start one of:
  llama-server -m gemma-X-Y.Q4_K_M.gguf -c 8192 --port 8080 --host 0.0.0.0
  ollama serve  (then `ollama pull gemma2:2b` or whatever fits 8-11 GB)

Vocab caveat:
  GGUF teacher and HF student must share a tokenizer. Gemma 2 / 3 / 4
  variants share most of the Gemma vocab but Gemma 3 added multimodal
  tokens. A mismatch silently produces logprobs over the wrong vocab —
  the loss will look fine but the gradient signal is garbage. Pass
  --check-vocab to spot-check that the first few tokens round-trip
  through both vocabs.

Usage:
  uv run python data/local_generate_qa.py \\
    --base-url http://localhost:8080/v1 \\
    --model gemma-4-e2b-it-q4 \\
    --tokenizer google/gemma-4-E2B-it \\
    --ds-name cocoon_code_qa \\
    --split train \\
    --output-dir data/raw_datasets/cocoon_code_qa_self_gen
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from datasets import Dataset
from transformers import AutoTokenizer

# These templates match self_generate_qa.py's gemma branch — single user
# message (Gemma chat templates don't support a system role), with the
# system instruction prefixed onto the user content.
from ctx_to_lora.data.self_gen_template import (
    QA_PROMPT_TEMPLATE,
    SELF_GEN_SYSTEM_MSG,
)
from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer


def _build_user_content(context: str, question: str) -> str:
    """Mirror self_generate_qa.py:create_messages — gemma chat template
    can't carry a system role, so the system instruction goes inline."""
    inner = QA_PROMPT_TEMPLATE.format(context=context, question=question)
    return SELF_GEN_SYSTEM_MSG + "\n\n\n" + inner


def _post_chat(
    base_url: str,
    model: str,
    user_content: str,
    max_new_tokens: int,
    temperature: float,
    top_logprobs: int,
    timeout_s: float,
) -> dict:
    """One request to /v1/chat/completions. Returns the parsed JSON."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        # Greedy + deterministic.
        "seed": 42,
    }
    r = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()


def _decode_logprobs(
    resp_json: dict,
    tokenizer,
    top_k: int,
    unknown_token_id: int,
) -> tuple[str, np.ndarray, np.ndarray, list[int]] | None:
    """Pull (text, logprobs_vals[N,K], logprobs_indices[N,K], token_ids[N])
    out of an OpenAI-style chat-completion response.

    Returns None if the response is malformed or the chosen-token chain
    has any holes — KL distillation expects a contiguous per-token
    distribution, no skips.

    Token strings → token IDs round-trips through the local tokenizer.
    If a teacher token doesn't re-encode to a single ID we slot in
    `unknown_token_id` and log a warning; for a downstream KL loss this
    leaks a tiny amount of mass to <unk> but is otherwise harmless.
    """
    choice = (resp_json.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    logprobs_block = choice.get("logprobs") or {}
    per_token = logprobs_block.get("content") or []
    if not per_token:
        return None

    n = len(per_token)
    vals = np.full((n, top_k), -1e9, dtype=np.float16)
    indices = np.full((n, top_k), unknown_token_id, dtype=np.int32)
    chosen_ids: list[int] = []

    def _encode_single(tok_str: str) -> int:
        # add_special_tokens=False so we don't get a leading BOS.
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        # Multi-token fallback: best-effort, take the first ID and warn.
        if ids:
            return ids[0]
        return unknown_token_id

    for t, entry in enumerate(per_token):
        chosen_ids.append(_encode_single(entry.get("token") or ""))
        candidates = entry.get("top_logprobs") or []
        for k, c in enumerate(candidates[:top_k]):
            vals[t, k] = c.get("logprob", -1e9)
            indices[t, k] = _encode_single(c.get("token") or "")

    return text, vals, indices, chosen_ids


def _resolve_dataset_path(ds_name: str, split: str) -> Path:
    """cocoon_code_qa lives at data/raw_datasets/{ds_name}/{split}/ds.parquet
    after the orchestrator's convert step. Also accepts an absolute path."""
    if os.path.isabs(ds_name) or ds_name.startswith("./"):
        return Path(ds_name) / split / "ds.parquet"
    # Match definitions.RAW_DATA_DIR convention.
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "data" / "raw_datasets" / ds_name / split / "ds.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate distillation-target answers + logprobs "
        "from a llama.cpp / Ollama server over the OpenAI-compat API.",
    )
    ap.add_argument("--base-url", default="http://localhost:8080/v1",
                    help="OpenAI-compat endpoint. llama-server defaults to "
                    "http://localhost:8080/v1; Ollama serves at "
                    "http://localhost:11434/v1.")
    ap.add_argument("--model", required=True,
                    help="Model id the server expects (llama.cpp ignores it; "
                    "Ollama needs the model tag, e.g. 'gemma2:2b').")
    ap.add_argument("--tokenizer", required=True,
                    help="HF tokenizer id used to re-encode logprobs back to "
                    "vocab IDs. MUST match the downstream student model — "
                    "for d2l Gemma 4 E2B training that's "
                    "google/gemma-4-E2B-it.")
    ap.add_argument("--ds-name", required=True,
                    help="Dataset name under data/raw_datasets/ (e.g. "
                    "'cocoon_code_qa') or an absolute path to a raw_datasets "
                    "subdirectory.")
    ap.add_argument("--split", default="train",
                    help="Dataset split. Default: train.")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Where to write the {split}/ds.parquet output. "
                    "Convention: data/raw_datasets/<ds_name>_self_gen.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-logprobs", type=int, default=16,
                    help="K in top-K logprobs per generated token. Matches "
                    "self_generate_qa.py's hardcoded k=16. OpenAI caps at "
                    "20; llama.cpp tends to cap around there too.")
    ap.add_argument("--timeout-s", type=float, default=300.0,
                    help="Per-request timeout. Generous default for slow "
                    "Pascal-class GPUs running Q4 GGUFs.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N (context, question) pairs. "
                    "0 = no limit. Use for smoke-testing the server.")
    ap.add_argument("--check-vocab", action="store_true",
                    help="Before any real work, send one trivial prompt and "
                    "verify each returned top-1 token round-trips through "
                    "the local tokenizer. Catches vocab mismatches up front.")
    args = ap.parse_args()

    ds_path = _resolve_dataset_path(args.ds_name, args.split)
    if not ds_path.exists():
        sys.exit(f"dataset parquet not found: {ds_path}")

    print(f"[gen] loading {ds_path}", flush=True)
    ds = Dataset.from_parquet(str(ds_path))
    print(f"[gen] {len(ds)} rows; columns: {ds.column_names}", flush=True)

    if "context" not in ds.column_names or "prompts" not in ds.column_names:
        sys.exit(
            "dataset is missing 'context' / 'prompts' columns — this script "
            "expects the cocoon_code_qa schema."
        )

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # Same tokenizer call internalize() uses, for byte-identical ctx_ids
    # between this generation pass and downstream training.
    ctx_tok = get_tokenizer(args.tokenizer)
    unk_id = tok.unk_token_id if tok.unk_token_id is not None else 0

    if args.check_vocab:
        print("[gen] check-vocab: pinging server with a trivial prompt...",
              flush=True)
        probe = _post_chat(
            args.base_url, args.model,
            "Reply with the single word: hello.",
            max_new_tokens=4, temperature=0.0, top_logprobs=1,
            timeout_s=args.timeout_s,
        )
        decoded = _decode_logprobs(probe, tok, top_k=1,
                                   unknown_token_id=unk_id)
        if decoded is None:
            sys.exit("vocab check failed: server response had no logprobs.")
        text, _, _, ids = decoded
        roundtrip_ok = all(
            tok.encode(per["token"], add_special_tokens=False)[:1] == [ids[i]]
            for i, per in enumerate(probe["choices"][0]["logprobs"]["content"])
        )
        print(f"[gen] check-vocab: server returned {text!r}; "
              f"round-trip {'OK' if roundtrip_ok else 'MISMATCH — vocab differs!'}",
              flush=True)
        if not roundtrip_ok:
            sys.exit(
                "Vocab mismatch between server's tokenizer and --tokenizer. "
                "Distillation training against this output will compute KL "
                "over the wrong space. Use a GGUF built from the same HF "
                "checkpoint as --tokenizer."
            )

    out_rows: list[dict] = []
    n_done = 0
    n_skips = 0
    t_start = time.time()

    for row in ds:
        context = row["context"]
        prompts = row["prompts"] or []
        if not prompts:
            continue

        # Tokenize context once per row — matches what training does.
        ctx_ids = tokenize_ctx_text({"context": [context]}, ctx_tok)["ctx_ids"]

        per_question_results: list[dict] = []
        for q in prompts:
            if args.limit and n_done >= args.limit:
                break

            user_content = _build_user_content(context, q)
            # Pre-encode the prompt portion so we can compute
            # response_start_end without round-tripping the whole convo.
            prompt_ids = tok.apply_chat_template(
                [{"role": "user", "content": user_content}],
                add_generation_prompt=True,
                tokenize=True,
            )
            try:
                resp = _post_chat(
                    args.base_url, args.model, user_content,
                    args.max_new_tokens, args.temperature,
                    args.top_logprobs, args.timeout_s,
                )
            except requests.RequestException as e:
                print(f"[gen] server error on q={q[:60]!r}: {e}",
                      file=sys.stderr, flush=True)
                n_skips += 1
                n_done += 1
                continue

            decoded = _decode_logprobs(resp, tok,
                                       top_k=args.top_logprobs,
                                       unknown_token_id=unk_id)
            if decoded is None:
                print(f"[gen] empty logprobs on q={q[:60]!r}",
                      file=sys.stderr, flush=True)
                n_skips += 1
                n_done += 1
                continue
            text, vals, indices, chosen_ids = decoded

            input_ids = list(prompt_ids) + list(chosen_ids)
            response_start_end = (len(prompt_ids), len(input_ids))

            per_question_results.append({
                "input_ids": np.asarray(input_ids, dtype=np.int32),
                "response_start_end": np.asarray(response_start_end,
                                                 dtype=np.int32),
                "logprobs_vals": vals,
                "logprobs_indices": indices,
                "response_text": text,
            })
            n_done += 1

            if n_done % 10 == 0:
                rate = n_done / max(1.0, time.time() - t_start)
                print(f"[gen] {n_done} qs done ({rate:.2f}/s, "
                      f"{n_skips} skipped)", flush=True)

        if not per_question_results:
            continue

        # One output row per context, vector-of-per-question columns. Matches
        # self_generate_qa.py:447-454.
        out_rows.append({
            "context": context,
            "ctx_ids": np.asarray(ctx_ids[0], dtype=np.int32),
            "input_ids": [r["input_ids"] for r in per_question_results],
            "response_start_end": [r["response_start_end"]
                                   for r in per_question_results],
            "logprobs_vals": [r["logprobs_vals"]
                              for r in per_question_results],
            "logprobs_indices": [r["logprobs_indices"]
                                 for r in per_question_results],
            # Bonus: keep the raw text so you can sanity-check answers.
            "responses": [r["response_text"]
                          for r in per_question_results],
        })

        if args.limit and n_done >= args.limit:
            break

    if not out_rows:
        sys.exit("no rows generated — check server is up and --ds-name is right.")

    out_dir = args.output_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ds.parquet"
    print(f"[gen] writing {len(out_rows)} contexts -> {out_path}", flush=True)
    Dataset.from_list(out_rows).to_parquet(str(out_path))
    print(f"[gen] done in {time.time() - t_start:.1f}s; "
          f"{n_done} qs ({n_skips} skipped).", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
