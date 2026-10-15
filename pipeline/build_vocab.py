"""Count the words of the train split and keep the frequent ones as word features, as fastText's dictionary does."""

import json
import os

from argparse import ArgumentParser
from collections import Counter
from glob import glob
from multiprocessing import Pool

from src._config import DEFAULT_N_THREADS, DEFAULT_SPLIT_DIR
from src.lid.features import tokenize

# Rarer words get no feature of their own, only their character n-grams: fastText's -minCount, as NLLB, OpenLID,
# GlotLID and ConLID set it
MIN_COUNT = 1000


def count_words(class_dir):
    counts = Counter()
    for path in glob(f"{class_dir}/*.txt"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                counts.update(tokenize(line))
    return counts


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-threads",
        type=int,
        default=DEFAULT_N_THREADS,
        help="CPU processes counting classes in parallel",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_file = f"{DEFAULT_SPLIT_DIR}/vocab.json"
    if os.path.exists(output_file):
        raise SystemExit(f"{output_file} exists: move or remove it first")

    counts = Counter()
    with Pool(args.n_threads) as pool:
        for class_counts in pool.imap_unordered(
            count_words, glob(f"{DEFAULT_SPLIT_DIR}/train/*")
        ):
            counts.update(class_counts)
    # Most frequent first, as fastText orders its dictionary (ties by word): a word's position is its feature id
    frequent = sorted(
        ((word, n) for word, n in counts.items() if n >= MIN_COUNT),
        key=lambda item: (-item[1], item[0]),
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dict(frequent), f, ensure_ascii=False, indent=0)
    print(
        f"{len(counts)} words, {len(frequent)} seen >= {MIN_COUNT} times -> {output_file}"
    )


if __name__ == "__main__":
    main()
