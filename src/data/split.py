"""Read, deduplicate and split a GlotLID-C label's sentences into train, dev and test.

Adapted from GlotLID's dedup_split.py: https://github.com/cisnlp/GlotLID/blob/main/assets/train/dedup_split.py
"""

from pathlib import Path

# Share of a label's sentences per split (GlotLID's dedup_split.py), then a cap per label: dev and test as the GlotLID
# paper's test set, train as ConLID
SHARES = {"dev": 0.1, "test": 0.1}
CAPS = {"train": 100_000, "dev": 1_000, "test": 1_000, "heldout": 1_000}


def read_sentences(label_dir):
    """A label's lines as (source, sentence), stripped, without empty lines or duplicates across its files."""
    label = Path(label_dir).name
    seen = set()
    sentences = []
    # Files <label>_<source>.txt in name order: a duplicate is kept in the first source
    for path in sorted(Path(label_dir).glob("*.txt")):
        source = path.stem.removeprefix(f"{label}_")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    sentences.append((source, line))
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


def write_sentences(sentences, label_dir):
    """Write (source, sentence) pairs as one file per source, <label>_<source>.txt as in GlotLID-C."""
    label = Path(label_dir).name
    by_source = {}
    for source, sentence in sentences:
        by_source.setdefault(source, []).append(sentence)
    Path(label_dir).mkdir(parents=True, exist_ok=True)
    for source, lines in by_source.items():
        with open(f"{label_dir}/{label}_{source}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
