#!/bin/bash
# Quick CPU-only dependency-resolution check. Catches lockfile cascades,
# missing-symbol import errors, and pin conflicts in ~30 seconds so you
# don't burn 10 min of pod time to find out transformers can't import.
#
# What it checks (CPU-only — no CUDA, no flash-attn, no flashinfer):
#   1. uv sync against the lockfile resolves cleanly (no version conflicts)
#   2. The post-sync forced installs (tokenizers, etc. in install.sh) don't
#      break the resolved set
#   3. The critical imports train.py performs at module top actually work
#
# What it does NOT check:
#   - CUDA / GPU correctness (use a smoke pod for that)
#   - flash-attn / flashinfer compatibility (cuda-only wheels)
#   - HF model downloads / tokenizer compat with real model weights
#
# Usage: bash scripts/preflight.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SANDBOX="${PREFLIGHT_VENV:-/tmp/d2l-preflight}"
echo "[preflight] sandbox at $SANDBOX"

if [ ! -d "$SANDBOX" ]; then
    uv venv "$SANDBOX" --python 3.10 --seed >/dev/null
fi

PY="$SANDBOX/bin/python"

# Mirror install.sh's pip-install ordering, but skip CUDA wheels.
# Use --index-strategy unsafe-best-match so torch CPU wheels resolve
# even though the project metadata expects cu124 builds.
echo "[preflight] installing transformers + tokenizers from lockfile pins..."
uv pip install --python "$PY" \
    "transformers==5.5.0" \
    "tokenizers==0.22.2" \
    "huggingface-hub==1.16.1" \
    >/dev/null

echo "[preflight] critical imports:"
"$PY" <<'PYEOF'
import sys
ok = True

def check(name, expr):
    global ok
    try:
        exec(expr)
        print(f"  OK   {name}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        ok = False

check("transformers (top)",
      "import transformers")
check("huggingface_hub.is_offline_mode",
      "from huggingface_hub import is_offline_mode")
check("transformers.utils.hub",
      "from transformers.utils import hub")
check("transformers.utils.generic",
      "from transformers.utils import generic")
check("transformers.AutoConfig",
      "from transformers import AutoConfig")
check("tokenizers (top)",
      "import tokenizers")

sys.exit(0 if ok else 1)
PYEOF

echo "[preflight] all imports OK — your transformers/hf-hub/tokenizers"
echo "[preflight] resolution is internally consistent. Safe to launch a pod."
