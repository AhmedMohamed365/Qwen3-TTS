#!/usr/bin/env python3
# =============================================================================
# prepare_speaker_data.py
#
# Reads train.csv from a Kaggle input dataset, filters rows for a target
# speaker (default: "Speaker1"), copies the matching audio segments to
# /kaggle/working/speaker_data/, and preprocesses them to mono 24 kHz WAV.
#
# Usage (from a Kaggle notebook cell):
#   !source ./miniconda3/bin/activate qwen_tts_env && \
#       python scripts/prepare_speaker_data.py \
#           --csv_path /kaggle/input/<dataset>/train.csv \
#           --audio_root /kaggle/input/<dataset> \
#           --output_dir /kaggle/working/speaker_data \
#           --speaker Speaker1
# =============================================================================
import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd


def preprocess_audio(src: str, dst: str, target_sr: int = 24000) -> None:
    """Convert *src* to mono, 24 kHz WAV and write to *dst*.

    Uses ``ffmpeg`` which is available on Kaggle after system-dep install.
    Falls back to ``sox`` if ffmpeg is not found.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Prefer ffmpeg – it is faster and widely available on Kaggle
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src,
                "-ac", "1",           # mono
                "-ar", str(target_sr), # 24 kHz
                "-sample_fmt", "s16",  # 16-bit PCM
                dst,
            ],
            check=True,
            capture_output=True,
        )
        return
    except FileNotFoundError:
        pass  # ffmpeg not on PATH – try sox

    # Fallback: sox
    subprocess.run(
        ["sox", src, "-r", str(target_sr), "-c", "1", "-b", "16", dst],
        check=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter train.csv for a target speaker, copy & preprocess audio.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/kaggle/input/train.csv",
        help="Path to the train.csv file.",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default="/kaggle/input",
        help="Root directory that contains the WAV files referenced in FileName.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/kaggle/working/speaker_data",
        help="Directory where preprocessed WAV files will be written.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Speaker1",
        help="Target speaker label (substring match against the Speaker column).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    print(f"▶ Loading CSV from {args.csv_path} …")
    df = pd.read_csv(args.csv_path)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    expected_cols = {"FileName", "Speaker"}
    if not expected_cols.issubset(set(df.columns)):
        print(
            f"ERROR: CSV must contain columns {expected_cols}. "
            f"Found: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Filter for target speaker
    # ------------------------------------------------------------------
    mask = df["Speaker"].astype(str).str.contains(args.speaker, case=False, na=False)
    speaker_df = df[mask].copy()
    print(f"✓ Found {len(speaker_df)} rows for speaker '{args.speaker}' "
          f"(out of {len(df)} total rows).")

    if speaker_df.empty:
        print("WARNING: No rows matched – nothing to do.", file=sys.stderr)
        sys.exit(0)

    # ------------------------------------------------------------------
    # 3. Copy & preprocess each audio file
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    processed_files: list[str] = []
    skipped = 0
    for _, row in speaker_df.iterrows():
        rel_path = str(row["FileName"]).strip()
        src_path = os.path.join(args.audio_root, rel_path)

        if not os.path.isfile(src_path):
            print(f"  ⚠ Source file not found, skipping: {src_path}")
            skipped += 1
            continue

        # Flatten into output_dir keeping only the base filename
        dst_name = os.path.basename(rel_path)
        dst_path = os.path.join(args.output_dir, dst_name)

        try:
            preprocess_audio(src_path, dst_path, target_sr=24000)
        except subprocess.CalledProcessError as exc:
            print(
                f"  ⚠ Failed to preprocess {src_path}: {exc}. "
                "Common causes: corrupted audio file or missing codec support.",
            )
            skipped += 1
            continue
        processed_files.append(dst_path)

    print(f"✓ Preprocessed {len(processed_files)} files → {args.output_dir}")
    if skipped:
        print(f"  ⚠ Skipped {skipped} files (not found on disk).")


if __name__ == "__main__":
    main()
