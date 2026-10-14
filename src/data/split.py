"""Deduplicate a class's sentences and split them into train, dev and test.

Adapted from GlotLID's dedup_split.py: https://github.com/cisnlp/GlotLID/blob/main/assets/train/dedup_split.py
"""

from pathlib import Path

# Share of a class's sentences per split (GlotLID's dedup_split.py), then a cap per class: dev and test as the GlotLID
# paper's test set, train as ConLID
SHARES = {"dev": 0.1, "test": 0.1}
CAPS = {"train": 100_000, "dev": 1_000, "test": 1_000, "heldout": 1_000}


def read_lines(paths):
    """Lines of text files as (file name without extension, line), in the order of paths."""
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield Path(path).stem, line


def unique_sentences(pairs):
    """(name, sentence) pairs, stripped, without empty sentences or duplicates: a duplicate is kept under its first name."""
    seen = set()
    sentences = []
    for name, sentence in pairs:
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            sentences.append((name, sentence))
    return sentences


def split_sentences(sentences):
    """Shuffled sentences -> train, dev and test by SHARES (train gets the rest), each capped at CAPS."""
    n_dev = int(len(sentences) * SHARES["dev"])
    n_test = int(len(sentences) * SHARES["test"])
    splits = {
        "test": sentences[:n_test],
        "dev": sentences[n_test : n_test + n_dev],
        "train": sentences[n_test + n_dev :],
    }
    return {split: rows[: CAPS[split]] for split, rows in splits.items()}


def write_sentences(sentences, out_dir):
    """Write (name, sentence) pairs as one file per name, <out_dir>/<name>.txt."""
    by_name = {}
    for name, sentence in sentences:
        by_name.setdefault(name, []).append(sentence)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, lines in by_name.items():
        with open(f"{out_dir}/{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
