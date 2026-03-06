#!/usr/bin/env python3
# =============================================================================
# create_finetune_jsonl.py
#
# Generates the train_raw.jsonl required by the Qwen3-TTS finetuning pipeline.
#
# Reads the original train.csv to get the transcript (ShowName column) for each
# Speaker1 segment, then pairs it with the preprocessed WAV in --audio_dir and
# a single reference audio file (--ref_audio).
#
# Output: a JSONL file where each line is:
#   {"audio": "<path>", "text": "<transcript>", "ref_audio": "<ref_path>"}
#
# Usage:
#   python scripts/create_finetune_jsonl.py \
#       --csv_path  /kaggle/input/<dataset>/train.csv \
#       --audio_dir /kaggle/working/speaker_data \
#       --ref_audio /kaggle/working/ref_audio/ref.wav \
#       --output    /kaggle/working/train_raw.jsonl \
#       --speaker   Speaker1
# =============================================================================
import argparse
import json
import os
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train_raw.jsonl for Qwen3-TTS finetuning.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/kaggle/input/train.csv",
        help="Path to the original train.csv.",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/kaggle/working/speaker_data",
        help="Directory with preprocessed (mono 24 kHz) WAV files.",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        default="/kaggle/working/ref_audio/ref.wav",
        help="Path to the reference speaker audio (same for all samples).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/kaggle/working/train_raw.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Speaker1",
        help="Target speaker label (substring match against the Speaker column).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load & filter CSV
    # ------------------------------------------------------------------
    print(f"▶ Loading CSV from {args.csv_path} …")
    df = pd.read_csv(args.csv_path)
    df.columns = df.columns.str.strip()

    required = {"FileName", "ShowName", "Speaker"}
    if not required.issubset(set(df.columns)):
        print(
            f"ERROR: CSV must contain columns {required}. Found: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    mask = df["Speaker"].astype(str).str.contains(args.speaker, case=False, na=False)
    speaker_df = df[mask].copy()
    print(f"✓ {len(speaker_df)} rows matched speaker '{args.speaker}'.")

    if speaker_df.empty:
        print("WARNING: No rows matched – nothing to write.", file=sys.stderr)
        sys.exit(0)

    # ------------------------------------------------------------------
    # 2. Build JSONL entries
    # ------------------------------------------------------------------
    entries: list[dict] = []
    skipped = 0
    for _, row in speaker_df.iterrows():
        base_name = os.path.basename(str(row["FileName"]).strip())
        audio_path = os.path.join(args.audio_dir, base_name)

        if not os.path.isfile(audio_path):
            skipped += 1
            continue

        text = str(row["ShowName"]).strip()
        if not text:
            skipped += 1
            continue

        entries.append({
            "audio": audio_path,
            "text": text,
            "ref_audio": args.ref_audio,
        })

    # ------------------------------------------------------------------
    # 3. Write JSONL
    # ------------------------------------------------------------------
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {len(entries)} entries → {args.output}")
    if skipped:
        print(f"  ⚠ Skipped {skipped} rows (audio not found or empty text).")


if __name__ == "__main__":
    main()
