#!/usr/bin/env bash
# =============================================================================
# run_pipeline_kaggle.sh
#
# End-to-end pipeline for Kaggle:
#   1. Filter & preprocess Speaker1 audio from train.csv
#   2. Create train_raw.jsonl
#   3. Encode audio codes  (prepare_data.py)
#   4. Fine-tune the model (sft_12hz.py)
#
# Prerequisites:
#   • initialize_kaggle.sh has been run (conda env ready)
#   • Reference audio placed at REF_AUDIO path (see below)
#   • Kaggle dataset mounted at /kaggle/input/<DATASET_NAME>
#
# Usage (from a Kaggle notebook cell):
#   !bash scripts/run_pipeline_kaggle.sh
#
# All paths are configurable via environment variables (see defaults below).
# =============================================================================
set -euo pipefail

# ── Download code to /kaggle/working ────────────────────────────────────────────
# mkdir -p /kaggle/working
# git clone https://github.com/QwenLM/Qwen3-TTS.git /kaggle/working/Qwen3-TTS

# ── Configurable paths ───────────────────────────────────────────────────────
ROOT_DIR="${ROOT_DIR:-/kaggle/working/Qwen3-TTS}"
ENV_NAME="${ENV_NAME:-qwen_tts_env}"
DATASET_NAME="${DATASET_NAME:-datasets/sdaiancai/sada2022}"

CSV_PATH="${CSV_PATH:-/kaggle/input/${DATASET_NAME}/train.csv}"
AUDIO_ROOT="${AUDIO_ROOT:-/kaggle/input/${DATASET_NAME}}"
SPEAKER="${SPEAKER:-Speaker1}"

# ── processed audio + JSONL intermediates → /kaggle/temp (saves /kaggle/working space) ──
SPEAKER_DATA_DIR="${SPEAKER_DATA_DIR:-/kaggle/temp/processed_data}"
RAW_JSONL="${RAW_JSONL:-/kaggle/temp/train_raw.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-/kaggle/temp/train_with_codes.jsonl}"

REF_AUDIO="${REF_AUDIO:-/kaggle/working/ref_audio/ref.wav}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/output}"

DEVICE="${DEVICE:-cuda:0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-TTS-Tokenizer-12Hz}"
INIT_MODEL="${INIT_MODEL:-/kaggle/working/Qwen3-TTS/Qwen/Qwen3-TTS-12Hz-0.6B-Base}"

NUM_SAMPLES="${NUM_SAMPLES:-500}"
MAX_DURATION="${MAX_DURATION:-15}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-100}"
SPEAKER_NAME="${SPEAKER_NAME:-speaker1}"
REF_TEXT="${REF_TEXT:-مرحبا بكم في هذا البرنامج}"

# ── Resolve conda python ─────────────────────────────────────────────────────
PYTHON="$ROOT_DIR/miniconda3/envs/$ENV_NAME/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: conda env '$ENV_NAME' not found. Run initialize_kaggle.sh first." >&2
    exit 1
fi

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
info() { echo -e "${YELLOW}▶ $*${NC}"; }

# ── Ensure /kaggle/temp directories exist ────────────────────────────────────
mkdir -p "$SPEAKER_DATA_DIR"
mkdir -p "$(dirname "$RAW_JSONL")"
mkdir -p "$(dirname "$TRAIN_JSONL")"

# ── Step 1: Filter & preprocess Speaker1 audio ──────────────────────────────
info "Step 1 / 4 — Filtering & preprocessing audio for '${SPEAKER}' …"
"$PYTHON" "$ROOT_DIR/scripts/prepare_speaker_data.py" \
    --csv_path      "$CSV_PATH"         \
    --audio_root    "$AUDIO_ROOT"       \
    --output_dir    "$SPEAKER_DATA_DIR" \
    --speaker       "$SPEAKER"          \
    --num_samples   "$NUM_SAMPLES"      \
    --max_duration  "$MAX_DURATION"
ok "Speaker audio ready at $SPEAKER_DATA_DIR"

# ── Step 2: Create train_raw.jsonl ──────────────────────────────────────────
info "Step 2 / 4 — Creating train_raw.jsonl …"

if [[ ! -f "$REF_AUDIO" ]]; then
    echo ""
    echo "⚠  Reference audio not found at: $REF_AUDIO"
    echo "   Please place your reference WAV there before running this step."
    echo "   Example:"
    echo "     mkdir -p /kaggle/working/ref_audio"
    echo "     cp /kaggle/input/<dataset>/some_ref.wav $REF_AUDIO"
    echo ""
    exit 1
fi

"$PYTHON" "$ROOT_DIR/scripts/create_finetune_jsonl.py" \
    --csv_path  "$CSV_PATH"         \
    --audio_dir "$SPEAKER_DATA_DIR" \
    --ref_audio "$REF_AUDIO"        \
    --output    "$RAW_JSONL"        \
    --speaker   "$SPEAKER"
ok "JSONL created at $RAW_JSONL"

# ── Step 3: Encode audio codes ──────────────────────────────────────────────
info "Step 3 / 4 — Encoding audio codes (prepare_data.py) …"
"$PYTHON" "$ROOT_DIR/finetuning/prepare_data.py" \
    --device "$DEVICE" \
    --tokenizer_model_path "$TOKENIZER_MODEL" \
    --input_jsonl  "$RAW_JSONL" \
    --output_jsonl "$TRAIN_JSONL" \
    --batch_infer_num 1
ok "Audio codes encoded → $TRAIN_JSONL"

# ── Step 4: Fine-tune ───────────────────────────────────────────────────────
info "Step 4 / 4 — Fine-tuning the model …"
mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR/finetuning"
"$PYTHON" sft_12hz.py \
    --init_model_path   "$INIT_MODEL"   \
    --output_model_path "$OUTPUT_DIR"   \
    --train_jsonl       "$TRAIN_JSONL"  \
    --batch_size        "$BATCH_SIZE"   \
    --lr                "$LR"           \
    --num_epochs        "$EPOCHS"       \
    --speaker_name      "$SPEAKER_NAME"
cd "$ROOT_DIR"
ok "Fine-tuning complete — checkpoints at $OUTPUT_DIR"

echo ""
echo "=========================================="
echo " Disk usage summary:"
du -sh "$SPEAKER_DATA_DIR" 2>/dev/null | awk '{print "   processed_data : "$1}'
du -sh "$TRAIN_JSONL"      2>/dev/null | awk '{print "   train JSONL    : "$1}'
du -sh "$OUTPUT_DIR"       2>/dev/null | awk '{print "   model output   : "$1}'
echo " All done!  Checkpoints saved to:"
echo "   $OUTPUT_DIR/checkpoint-epoch-*"
echo ""
echo " Quick inference test:"
echo "   source $ROOT_DIR/miniconda3/bin/activate $ENV_NAME"
echo "   python -c \""
echo "     import torch, soundfile as sf"
echo "     from qwen_tts import Qwen3TTSModel"
echo "     tts = Qwen3TTSModel.from_pretrained("
echo "         '$OUTPUT_DIR/checkpoint-epoch-$((EPOCHS-1))',"
echo "         device_map='$DEVICE',"
echo "         dtype=torch.bfloat16,"
echo "     )"
echo "     wavs, sr = tts.generate_custom_voice("
echo "         text='Hello world',"
echo "         speaker='$SPEAKER_NAME',"
echo "     )"
echo "     sf.write('output.wav', wavs[0], sr)"
echo "   \""
echo "=========================================="
