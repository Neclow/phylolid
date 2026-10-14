"""Pick the held-out languages of GlotLID-C and Omnilingual, and split GlotLID-C into train, dev, test and held-out sentences."""

import random

from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from ete4 import Tree
from pyglottolog import Glottolog

from src._config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_EXTERN_DIR,
    DEFAULT_REFERENCE_TREE_PROCESSED_DIR,
    DEFAULT_SPLIT_DIR,
    HELDOUT_FAMILIES,
)
from src.data.split import CAPS, read_sentences, split_sentences, write_sentences
from src.models._model_zoo import MODEL_ZOO

GLOTTOLOG = f"{DEFAULT_EXTERN_DIR}/glottolog"
GLOTLID_DIR = f"{DEFAULT_DATASET_DIR}/glotlid-corpus/v3.1"
# Trained on GlotLID-C: a GlotLID-C language they know can still be held out (they get "--" on it)
TRAINED_ON_GLOTLID = ["cis-lmu/glotlid", "epfl-nlp/ConLID"]
# A GlotLID-C language can be held out with at least this many lines and a node in each of these reference trees
MIN_LINES = 1000
TREES = ["glottolog", "asjp21"]


def language_of(glottocode, languoids):
    """Glottolog language of a label: the languoid itself, or the language above a dialect; None for groups."""
    languoid = languoids.get(glottocode)
    if languoid is None or languoid.level.name not in ("language", "dialect"):
        return None
    if languoid.level.name == "language":
        return glottocode
    return [g for _, g, level in languoid.lineage if level.name == "language"][-1]


def known_languages(glottocodes, languoids, below):
    """Languages that model labels cover: a label's language, or every language below a group label."""
    known = set()
    for glottocode in glottocodes:
        language = language_of(glottocode, languoids)
        known |= {language} if language else below.get(glottocode, set())
    return known


def tree_nodes(path):
    # Every node of a processed reference tree is named by its glottocode, internal nodes included
    return {node.name for node in Tree(open(path), parser=1).traverse()}


def count_lines(label):
    return sum(
        path.read_bytes().count(b"\n")
        for path in Path(f"{GLOTLID_DIR}/{label}").glob("*.txt")
    )


def assign_roles(labels, models, languoids):
    """Role of each dataset label: train, heldout or dropped (GlotLID-C); heldout, known or unused (Omnilingual)."""
    below = {}
    for languoid in languoids.values():
        if languoid.level.name == "language":
            for _, glottocode, _ in languoid.lineage:
                below.setdefault(glottocode, set()).add(languoid.id)
    labels = labels.assign(
        language=labels["glottocode"].map(lambda g: language_of(g, languoids))
    )
    glotlid = labels["dataset"] == "glotlid-corpus"
    candidate = labels["H0"].isin(HELDOUT_FAMILIES) & labels["language"].notna()
    glotlid_languages = set(labels.loc[glotlid, "language"].dropna())
    omni_languages = set(labels.loc[~glotlid, "language"].dropna())

    # Omnilingual: missing from GlotLID-C and unknown to every model
    known_to_any = known_languages(models["glottocode"].dropna(), languoids, below)
    omni_heldout = set(
        labels.loc[~glotlid & candidate, "language"]
        .drop_duplicates()
        .loc[lambda s: ~s.isin(glotlid_languages | known_to_any)]
    )

    # GlotLID-C: unknown to every model not trained on GlotLID-C, not in Omnilingual, enough lines, in every tree
    other_models = models[~models["model_id"].isin(TRAINED_ON_GLOTLID)]
    known_to_others = known_languages(
        other_models["glottocode"].dropna(), languoids, below
    )
    candidates = labels[
        glotlid & candidate & ~labels["language"].isin(known_to_others | omni_languages)
    ]
    candidates = candidates.assign(lines=candidates["label"].map(count_lines))
    by_language = candidates.groupby("language").agg(
        lines=("lines", "sum"), glottocodes=("glottocode", set)
    )
    nodes = {
        tree: set().union(
            *(
                tree_nodes(
                    f"{DEFAULT_REFERENCE_TREE_PROCESSED_DIR}/{family}/{tree}.nwk"
                )
                for family in HELDOUT_FAMILIES
            )
        )
        for tree in TREES
    }
    # A language is in a tree if one of its labels' glottocodes is a node (a dialect label can be the node)
    in_trees = by_language["glottocodes"].map(
        lambda codes: all(codes & nodes[tree] for tree in TREES)
    )
    glotlid_heldout = set(
        by_language.index[(by_language["lines"] >= MIN_LINES) & in_trees]
    )

    def role(row):
        if row.dataset == "glotlid-corpus":
            # Family-level labels (nah_Latn, ...) have no Glottolog language
            if pd.isna(row.language):
                return "dropped"
            return "heldout" if row.language in glotlid_heldout else "train"
        if row.language in omni_heldout:
            return "heldout"
        # The GlotLID-C held-out languages exclude Omnilingual ones, so these are all training languages
        return "known" if row.language in glotlid_languages else "unused"

    return labels.assign(role=[role(row) for row in labels.itertuples()])


def split_label(task, seed, output_dir):
    label, role = task
    sentences = read_sentences(f"{GLOTLID_DIR}/{label}")
    # Seeded by label, so a label's split does not depend on the order the labels are processed in
    random.Random(f"{seed}-{label}").shuffle(sentences)
    if role == "train":
        splits = split_sentences(sentences)
    else:
        splits = {"heldout": sentences[: CAPS["heldout"]]}
    for split, rows in splits.items():
        write_sentences(rows, f"{output_dir}/{split}/{label}")
    return label, role, {split: len(rows) for split, rows in splits.items()}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed of the sentence shuffles"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Labels split in parallel; each worker holds a whole label in memory (rus_Cyrl: 9.2M lines)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if Path(DEFAULT_SPLIT_DIR).exists():
        raise SystemExit(
            f"{DEFAULT_SPLIT_DIR} exists: shelve it first (e.g. data/_splits)"
        )

    print("== Held-out languages ==")
    languoids = {l.id: l for l in Glottolog(GLOTTOLOG).languoids()}
    labels = pd.read_csv(f"{DEFAULT_DATASET_DIR}/glottolog.csv")
    models = pd.read_csv(f"{DEFAULT_CACHE_DIR}/glottolog.csv")
    assert set(models["model_id"]) == set(MODEL_ZOO), "rerun map_glottolog"
    labels = assign_roles(labels, models, languoids)
    Path(DEFAULT_SPLIT_DIR).mkdir(parents=True)
    columns = [
        "dataset",
        "label",
        "name",
        "glottocode",
        "level",
        "language",
        "H0",
        "role",
    ]
    labels[columns].to_csv(f"{DEFAULT_SPLIT_DIR}/languages.csv", index=False)
    print(pd.crosstab(labels["dataset"], labels["role"]).to_string())
    heldout = labels[labels["role"] == "heldout"].drop_duplicates(
        ["dataset", "language"]
    )
    family = heldout["H0"].map(lambda g: languoids[g].name)
    print(pd.crosstab(family, heldout["dataset"]).to_string())

    print("== Sentences ==")
    glotlid = labels[
        (labels["dataset"] == "glotlid-corpus") & (labels["role"] != "dropped")
    ]
    tasks = list(zip(glotlid["label"], glotlid["role"]))
    totals = {}
    with Pool(args.workers) as pool:
        for label, role, counts in pool.imap_unordered(
            partial(split_label, seed=args.seed, output_dir=DEFAULT_SPLIT_DIR), tasks
        ):
            print(f"{label} ({role}): {counts}")
            for split, n in counts.items():
                totals[split] = totals.get(split, 0) + n
    print(f"{len(tasks)} labels -> {DEFAULT_SPLIT_DIR}: {totals}")


if __name__ == "__main__":
    main()
