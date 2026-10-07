"""Download the transcripts of the Omnilingual ASR corpus from Hugging Face and split them into sentences."""

import re

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from huggingface_hub import HfFileSystem

from src._config import DEFAULT_DATASET_DIR

REPO = "facebook/omnilingual-asr-corpus"
# 2025-11-14
REVISION = "8648ba8946377697b427ae952076e49fc0e5e44d"
# The release has no version or tag of its own: the revision's short hash
VERSION = REVISION[:7]
OUTDIR = f"{DEFAULT_DATASET_DIR}/omnilingual-asr/{VERSION}"
# Text columns only: parquet stores each column separately, so the audio (bytes, path) is never fetched. The prompts,
# in prompt languages, are dropped
COLUMNS = [
    "raw_text",
    "speaker_id",
    "segment_id",
    "prompt_id",
    "duration",
    "iso_639_3",
    "glottocode",
    "iso_15924",
]
# Transcription tags for non-speech, removed from the text
TAGS = re.compile(r"<(laugh|hesitation|noise)>")
# Tag for words the transcriber left out: sentences that contain it are dropped
UNINTELLIGIBLE = "<unintelligible>"
# Sentence punctuation followed by a space, in the corpus' scripts: Latin and others (.!?), Arabic (؟ ۔), Devanagari
# and Odia (। ॥), Tibetan (། ༎)
SENTENCE_END = re.compile(r"(?<=[.!?؟۔।॥།༎]) ")
MAX_CHARS = 200
MIN_CHARS = 30


def split_sentences(text):
    """Sentences of a transcript: split at sentence punctuation, then cut at the last space before MAX_CHARS."""
    text = re.sub(r"\s+", " ", TAGS.sub("", text)).strip()
    sentences = []
    for piece in SENTENCE_END.split(text):
        while len(piece) > MAX_CHARS:
            cut = piece.rfind(" ", 0, MAX_CHARS + 1)
            # No space: cut after the last tsheg (་), which Tibetan writes after every syllable instead of spaces;
            # without one either, cut at MAX_CHARS
            if cut == -1:
                cut = piece.rfind("་", 0, MAX_CHARS) + 1 or MAX_CHARS
            sentences.append(piece[:cut])
            piece = piece[cut:].lstrip()
        sentences.append(piece)
    return [
        sentence
        for sentence in sentences
        if UNINTELLIGIBLE not in sentence and len(sentence) >= MIN_CHARS
    ]


def main():
    fs = HfFileSystem()
    # Folders data/<label>/ (labels <ISO 639-3>_<script>), each with train, dev and test parquet files
    for label_dir in sorted(fs.glob(f"datasets/{REPO}@{REVISION}/data/*")):
        label = Path(label_dir).name
        splits = []
        for path in sorted(fs.glob(f"{label_dir}/*.parquet")):
            with fs.open(path, "rb", block_size=1 << 20) as f:
                df = pq.ParquetFile(f).read(columns=COLUMNS).to_pandas()
            # All splits are test data for us; the split is kept in a column (train-00000-of-00002.parquet -> train)
            df["split"] = Path(path).name.split("-")[0]
            splits.append(df)
        df = pd.concat(splits, ignore_index=True)
        rows = len(df)

        # One sentence per row; rows without any sentence left are dropped
        df["raw_text"] = df["raw_text"].map(split_sentences)
        df = df.explode("raw_text").dropna(subset="raw_text")
        df = df.rename(columns={"raw_text": "sentence"})

        Path(f"{OUTDIR}/{label}").mkdir(parents=True, exist_ok=True)
        df.to_csv(f"{OUTDIR}/{label}/{label}.tsv", sep="\t", index=False)
        print(f"{label}: {rows} rows -> {len(df)} sentences")
    print(f"{REPO}@{REVISION[:7]} -> {OUTDIR}")


if __name__ == "__main__":
    main()
