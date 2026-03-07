#!/usr/bin/env bash
# =============================================================================
# initialize_kaggle.sh
#
# Run once at the start of a Kaggle session to set up the environment.
# Installs Miniconda, creates a conda env, and installs all training deps.
#
# Usage (from a Kaggle notebook cell):
#   !bash scripts/initialize_kaggle.sh
#
# Subsequent commands must source the env:
#   !source ./miniconda3/bin/activate qwen_tts_env && python ...
# Or use run_pipeline_kaggle.sh which handles this automatically.
# =============================================================================
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-.}"
ENV_NAME="qwen_tts_env"
CONDA="$ROOT_DIR/miniconda3/bin/conda"
PIP="$ROOT_DIR/miniconda3/envs/$ENV_NAME/bin/pip"
PYTHON="$ROOT_DIR/miniconda3/envs/$ENV_NAME/bin/python"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
info() { echo -e "${YELLOW}▶ $*${NC}"; }

mkdir -p "$ROOT_DIR"

# ── 1. Install Miniconda (skip if already present) ────────────────────────────
if [[ ! -f "$ROOT_DIR/miniconda3/bin/conda" ]]; then
    info "Downloading Miniconda …"
    wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$ROOT_DIR/miniconda3" -f
    rm /tmp/miniconda.sh
    ok "Miniconda installed at $ROOT_DIR/miniconda3"
else
    ok "Miniconda already present — skipping install"
fi

# ── 2. Accept conda ToS (required for newer conda versions) ───────────────────
# Newer conda versions require explicit ToS acceptance before any channel can be
# used.  The || true fallback keeps the script working on older conda builds
# that do not have the `tos` sub-command.
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r   2>/dev/null || true

# ── 3. Create env (skip if exists) ───────────────────────────────────────────
if [[ ! -d "$ROOT_DIR/miniconda3/envs/$ENV_NAME" ]]; then
    info "Creating conda env '$ENV_NAME' (Python 3.11) …"
    "$CONDA" create --name "$ENV_NAME" python=3.11 -y
    ok "Env '$ENV_NAME' created"
else
    ok "Conda env '$ENV_NAME' already exists — skipping"
fi

# ── 4. Install system dependencies ───────────────────────────────────────────
info "Installing system deps (sox, ffmpeg) …"
apt-get update -qq && apt-get install -y -q sox libsox-fmt-all ffmpeg 2>/dev/null || true
ok "System deps ready"

# ── 5. Install PyTorch (CUDA 12.x) ───────────────────────────────────────────
info "Installing PyTorch (cu121) …"
"$PIP" install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
ok "PyTorch installed: $("$PYTHON" -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"

# ── 6. Install training dependencies ─────────────────────────────────────────
info "Installing training deps (transformers, peft, datasets, accelerate …) …"
"$PIP" install --quiet \
    "transformers>=4.45" \
    "peft>=0.10" \
    "datasets>=2.14" \
    "accelerate>=0.26" \
    "soundfile" \
    "torchaudio" \
    "tqdm" \
    "huggingface_hub" \
    "librosa" \
    "einops" \
    "sox" \
    "onnxruntime" \
    "safetensors" \
    "pandas"
# torchcodec: needed by newer datasets for Audio decoding — non-fatal if it fails
"$PIP" install --quiet torchcodec 2>/dev/null || true
ok "Training deps installed"

# ── 7. Install qwen-tts package (this repo) ──────────────────────────────────
info "Installing qwen-tts from local repo …"
"$PIP" install --quiet -e "$ROOT_DIR"
ok "qwen-tts installed"

info "Environment is ready.  Activate with:"
echo "  source $ROOT_DIR/miniconda3/bin/activate $ENV_NAME"
