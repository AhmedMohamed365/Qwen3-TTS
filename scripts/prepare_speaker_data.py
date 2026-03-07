#!/usr/bin/env python3
# =============================================================================
# prepare_speaker_data.py
#
# Reads train.csv from a Kaggle input dataset, filters rows for a target
# speaker (default: "Speaker1"), extracts the matching audio segments using
# SegmentStart / SegmentEnd columns, and converts them to mono 24 kHz WAV
# in /kaggle/working/speaker_data/.
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
import sys

import librosa
import pandas as pd
import soundfile as sf


def segment_filename(original_filename: str, start: float, end: float) -> str:
    """Return an output filename derived from the source file and segment times.

    >>> segment_filename("audio.wav", 1.234, 5.678)
    'audio_1234_5678.wav'
    """
    stem = os.path.splitext(os.path.basename(original_filename))[0]
    start_ms = int(round(float(start) * 1000))
    end_ms = int(round(float(end) * 1000))
    return f"{stem}_{start_ms}_{end_ms}.wav"


def extract_segment(
    src: str,
    dst: str,
    start: float,
    end: float,
    target_sr: int = 24000,
) -> None:
    """Extract a segment from *src* (start–end in seconds), convert to mono 24 kHz WAV.

    Uses ``librosa`` for loading/resampling and ``soundfile`` for writing.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    audio, sr = librosa.load(
        src, sr=None, mono=True,
        offset=start, duration=end - start,
    )

    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

    sf.write(dst, audio, target_sr, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter train.csv for a target speaker, extract & preprocess audio segments.",
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

    expected_cols = {"FileName", "Speaker", "SegmentStart", "SegmentEnd"}
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
    # 3. Extract & preprocess each audio segment
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

        start = float(row["SegmentStart"])
        end = float(row["SegmentEnd"])

        dst_name = segment_filename(rel_path, start, end)
        dst_path = os.path.join(args.output_dir, dst_name)

        try:
            extract_segment(src_path, dst_path, start, end, target_sr=24000)
        except Exception as exc:
            print(
                f"  ⚠ Failed to extract segment {src_path} "
                f"[{start}–{end}]: {exc}",
            )
            skipped += 1
            continue
        processed_files.append(dst_path)

    print(f"✓ Preprocessed {len(processed_files)} segments → {args.output_dir}")
    if skipped:
        print(f"  ⚠ Skipped {skipped} segments.")


if __name__ == "__main__":
    main()
