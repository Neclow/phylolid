"""Download or build reference trees of a language family (Glottolog, ASJP) whose nodes are a dataset's glottocodes."""

import json
import os
import subprocess

from abc import ABC, abstractmethod
from argparse import ArgumentParser

import pandas as pd
import requests

from ete4 import Tree
from pyglottolog import Glottolog

from src._config import (
    DATASET_ALIASES,
    DEFAULT_DATASET_DIR,
    DEFAULT_EXTERN_DIR,
    DEFAULT_REFERENCE_TREE_PROCESSED_DIR,
    DEFAULT_REFERENCE_TREE_RAW_DIR,
    HELDOUT_FAMILIES,
)

# ASJP v21.1 (git submodule) and Holman's asjp62x, compiled by scripts/post_install.sh
ASJP_DIR = f"{DEFAULT_EXTERN_DIR}/asjp"
ASJP62X = f"{DEFAULT_EXTERN_DIR}/asjp-software/asjp62x"
# MEGA-CC options: neighbour-joining tree from a distance matrix
NJ_OPTIONS = "src/phylo/nj_distances.mao"
# Jäger's maximum-likelihood world tree from ASJP v19, without Glottolog constraint, and its doculects
# (https://osf.io/a97sz/: data/worldtree_ml_free.tre, data/languages.csv)
ASJP19_TREE_URL = "https://osf.io/download/sbh4q/"
ASJP19_LANGUAGES_URL = "https://osf.io/download/w4jnf/"
# ASJP tags a word list with the glottocode of its ISO code, the language's, also when the list is a dialect that has
# its own label (e.g. Norwegian Bokmål, nob_Latn). Such lists, by name -> the dialect's glottocode
ASJP_DIALECTS = {
    "NORWEGIAN_BOKMAAL": "norw1259",
    "NORWEGIAN_BOKMAAL_2": "norw1259",
    # ASJP's only Nynorsk list, recorded in Toten (eastern Norway)
    "NORWEGIAN_NYNORSK_TOTEN": "norw1262",
    # Asante, a variety of Twi
    "TWI_ASANTE": "twii1234",
    "TWI_FANTE": "fant1241",
}


def download(url, path):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    # OSF has returned empty files with status 200
    if not response.content:
        raise ValueError(f"Empty download: {url}")
    with open(path, "wb") as f:
        f.write(response.content)


def read_lists(path):
    """Header and word lists ([(ASJP name, lines)]) of an ASJP lists.txt."""
    # latin-1 reads every byte as one character, so writing back with it keeps the file's bytes
    with open(path, encoding="latin-1") as f:
        lines = f.readlines()
    # The header ends with the two blank lines after the list of sound symbols
    end = (
        next(
            i
            for i in range(1, len(lines))
            if not lines[i - 1].strip() and not lines[i].strip()
        )
        + 1
    )
    word_lists = []
    for line in lines[end:]:
        # A word list starts with its name line, <ASJP name>{<classifications>}, the only line starting with
        # neither a digit (words) nor a blank (properties)
        if line.strip() and not (line[0].isdigit() or line[0].isspace()):
            word_lists.append((line.split("{")[0], [line]))
        elif line.strip():
            word_lists[-1][1].append(line)
    return lines[:end], word_lists


def unattested_if_malformed(line, name):
    """A word line without the words asjp62x cannot read, and XXX (not attested) if none is left."""
    # Word lines are <item number> <gloss>\t<comma-separated words> //
    if not (line[:1].isdigit() and "\t" in line):
        return line
    item, rest = line.split("\t", 1)
    words, separator, tail = rest.rstrip("\n").partition(" //")
    # asjp62x merges the two symbols before ~ (three before $) into one sound, and crashes when they are missing
    # (e.g. NGOMBE_CAR's "blood", ~E). % marks loanwords
    kept = [
        word
        for word in words.split(",")
        if not any(
            (c == "~" and i < 2) or (c == "$" and i < 3)
            for i, c in enumerate(word.replace(" ", "").lstrip("%"))
        )
    ]
    if len(kept) == len(words.split(",")):
        return line
    new = ",".join(kept).strip() or "XXX"
    print(
        f"  {name} {item}: {words.strip()} -> {new} (malformed words treated as unattested)"
    )
    return f"{item}\t{new}{separator}{tail}\n"


def first_leaves(tree, leaf_glottocode):
    """Glottocode -> its first leaf in the tree (as in phylaudio, when a languoid has several word lists)."""
    anchors = {}
    for leaf in tree.leaves():
        anchors.setdefault(leaf_glottocode.get(leaf.name), leaf)
    return anchors


class BaseTreeProcessor(ABC):
    """Download (or build) a raw tree, then name the nodes of a family's glottocodes and prune it to them."""

    def __init__(self, name, raw_file, glottocode):
        self.name = name
        self.raw_file = raw_file
        processed_dir = f"{DEFAULT_REFERENCE_TREE_PROCESSED_DIR}/{glottocode}"
        os.makedirs(DEFAULT_REFERENCE_TREE_RAW_DIR, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        self.processed_file = f"{processed_dir}/{name}.nwk"
        self.processed_args_file = f"{processed_dir}/{name}.json"

    def maybe_download(self, overwrite=False):
        if os.path.exists(self.raw_file) and not overwrite:
            print(f"Target file {self.raw_file} already exists. Skipping download.")
        else:
            print(f"Downloading {self.name} tree...")
            self.download()

    @abstractmethod
    def download(self):
        """Download or build the raw tree."""

    @abstractmethod
    def anchors(self, tree):
        """Glottocode -> its node in the raw tree."""

    def process(self, labels, process_args):
        tree = Tree(open(self.raw_file), parser=1)
        anchors = self.anchors(tree)
        # Labels of one glottocode (e.g. ace_Arab, ace_Latn) share its node. A glottocode without its own node (e.g. a
        # dialect without an ASJP word list) is absent, so that no node holds two languages
        kept = labels["glottocode"].isin(anchors)
        nodes = []
        for glottocode in labels.loc[kept, "glottocode"].unique():
            # ASJP leaves are named by word list
            anchors[glottocode].name = glottocode
            nodes.append(anchors[glottocode])
        # Keep the glottocodes' nodes, also internal ones (e.g. French, above Jerriais); other single-child nodes are
        # removed and their branch lengths summed
        tree.prune(nodes, preserve_branch_length=True)
        tree.write(outfile=self.processed_file, parser=1)
        with open(self.processed_args_file, "w", encoding="utf-8") as f:
            json.dump(process_args, f, indent=4)
        print(
            f"  {kept.sum()} labels ({len(nodes)} glottocodes) -> {self.processed_file}; "
            f"absent: {(~kept).sum()} {labels.loc[~kept, 'label'].tolist()}"
        )


class GlottologTreeProcessor(BaseTreeProcessor):
    """Glottolog tree of the family: nodes are glottocodes, every branch has length 1 (one Glottolog level)."""

    def __init__(self, family):
        super().__init__(
            "glottolog",
            f"{DEFAULT_REFERENCE_TREE_RAW_DIR}/glottolog_{family.id}.nwk",
            family.id,
        )
        self.family = family

    def download(self):
        with open(self.raw_file, "w", encoding="utf-8") as f:
            f.write(self.family.newick_node(template="{l.id}").newick + ";")

    def anchors(self, tree):
        return {node.name: node for node in tree.traverse()}


class ASJP19TreeProcessor(BaseTreeProcessor):
    """Jäger's ASJP v19 world tree (the method of Jäger 2018), as published: maximum likelihood, root arbitrary."""

    def __init__(self, glottocode):
        super().__init__(
            "asjp19", f"{DEFAULT_REFERENCE_TREE_RAW_DIR}/asjp19.nwk", glottocode
        )
        self.languages_file = f"{DEFAULT_REFERENCE_TREE_RAW_DIR}/asjp19_languages.csv"

    def download(self):
        download(ASJP19_TREE_URL, self.raw_file)
        download(ASJP19_LANGUAGES_URL, self.languages_file)

    def anchors(self, tree):
        languages = pd.read_csv(self.languages_file, keep_default_na=False)
        id_to_glottocode = (
            dict(zip(languages["ID"], languages["Glottocode"])) | ASJP_DIALECTS
        )
        # Leaves are <family>.<genus>.<doculect ID>
        return first_leaves(
            tree,
            {
                leaf.name: id_to_glottocode.get(leaf.name.split(".")[-1])
                for leaf in tree.leaves()
            },
        )


class ASJP21TreeProcessor(BaseTreeProcessor):
    """ASJP v21.1 tree of the family, built as the ASJP team builds its World Language Tree: LDND distances between
    all its word lists (asjp62x), neighbour-joining (MEGA), rooted at the midpoint."""

    def __init__(self, glottocode, family_name):
        super().__init__(
            "asjp21",
            f"{DEFAULT_REFERENCE_TREE_RAW_DIR}/asjp21_{glottocode}.nwk",
            glottocode,
        )
        self.family_name = family_name
        self.prefix = self.raw_file.removesuffix(".nwk")

    def download(self):
        languages = pd.read_csv(f"{ASJP_DIR}/cldf/languages.csv", keep_default_na=False)
        in_family = set(languages.loc[languages["Family"] == self.family_name, "Name"])
        header, word_lists = read_lists(f"{ASJP_DIR}/raw/lists.txt")
        family_lists = [
            (name, lines) for name, lines in word_lists if name in in_family
        ]

        # asjp62x input: lists.txt's header (40 items, >= 28 attested, no lists extinct before 1700, no loans), then
        # the family's word lists. Name lines become short IDs, since asjp62x keeps only their start
        ids = pd.DataFrame(
            {
                "id": [f"L{i}" for i in range(len(family_lists))],
                "Name": [n for n, _ in family_lists],
            }
        )
        with open(f"{self.prefix}.txt", "w", encoding="latin-1") as f:
            f.writelines(header)
            for id_, (name, lines) in zip(ids["id"], family_lists):
                f.write(f"{id_}\n")
                f.writelines(unattested_if_malformed(line, name) for line in lines[1:])
            # The input ends with a blank line
            f.write("\n")
        ids.merge(languages[["Name", "Glottocode"]]).to_csv(
            f"{self.prefix}.csv", index=False
        )
        print(f"  {len(family_lists)} {self.family_name} word lists")

        with open(f"{self.prefix}.txt") as fin, open(f"{self.prefix}.meg", "w") as fout:
            subprocess.run([ASJP62X], stdin=fin, stdout=fout, check=True)
        subprocess.run(
            [
                "megacc",
                "-a",
                NJ_OPTIONS,
                "-d",
                f"{self.prefix}.meg",
                "-o",
                f"{self.prefix}_mega",
                "-s",
                "-n",
            ],
            check=True,
        )

        tree = Tree(open(f"{self.prefix}_mega.nwk"), parser=1)
        # asjp62x writes LDND x 10,000 (percentages with the dot removed): back to LDND
        for node in tree.traverse():
            if node.dist is not None:
                node.dist /= 10000
        # Rooted at its midpoint, as the ASJP World Language Tree
        tree.set_midpoint_outgroup()
        tree.write(outfile=self.raw_file, parser=1)

    def anchors(self, tree):
        ids = pd.read_csv(f"{self.prefix}.csv", keep_default_na=False)
        glottocodes = [
            ASJP_DIALECTS.get(name, glottocode)
            for name, glottocode in zip(ids["Name"], ids["Glottocode"])
        ]
        return first_leaves(tree, dict(zip(ids["id"], glottocodes)))


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        help=f"folder in {DEFAULT_DATASET_DIR}/ or its alias, e.g. glotlid (glotlid-corpus)",
    )
    parser.add_argument(
        "-g",
        "--glottolog-dir",
        default=f"{DEFAULT_EXTERN_DIR}/glottolog",
        help="Path to glottolog folder",
    )
    parser.add_argument(
        "--glottocode",
        required=True,
        help="Glottolog languoid code for the root of the tree to extract, e.g. indo1319, "
        "or all for every family in HELDOUT_FAMILIES (src/_config.py)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing downloaded files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Labels of all datasets, with their glottocodes and lineages (from map_glottolog)
    dataset = DATASET_ALIASES.get(args.dataset, args.dataset)
    labels = pd.read_csv(f"{DEFAULT_DATASET_DIR}/glottolog.csv")
    labels = labels[labels["dataset"] == dataset]
    # Classes are Glottolog languages and dialects; family-level labels (e.g. nah_Latn) are left out
    is_class = labels["level"].isin(["language", "dialect"])

    glottolog = Glottolog(args.glottolog_dir)
    glottocodes = HELDOUT_FAMILIES if args.glottocode == "all" else [args.glottocode]
    for glottocode in glottocodes:
        family = glottolog.languoid(glottocode)
        if family is None:
            raise ValueError(f"{glottocode} is not a Glottolog languoid")
        in_family = labels["H0"] == glottocode
        excluded = labels.loc[in_family & ~is_class, "label"].tolist()
        family_labels = labels[in_family & is_class]
        print(
            f"{glottocode}: {len(family_labels)} labels; left out, not a Glottolog language or dialect: {excluded}"
        )

        processors = [
            GlottologTreeProcessor(family),
            ASJP19TreeProcessor(glottocode),
            ASJP21TreeProcessor(glottocode, family.name),
        ]
        for processor in processors:
            processor.maybe_download(overwrite=args.overwrite)
            print(f"Processing: {processor.name}...")
            processor.process(family_labels, {**vars(args), "glottocode": glottocode})


if __name__ == "__main__":
    main()
