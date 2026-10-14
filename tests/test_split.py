"""Unit tests for the sentence split in src/data/split.py."""

import random

from src.data.split import (
    CAPS,
    read_sentences,
    split_sentences,
    write_sentences,
)


def test_read_sentences_deduplicates_across_files(tmp_path):
    label_dir = tmp_path / "abc_Latn"
    label_dir.mkdir()
    (label_dir / "abc_Latn_cbc.txt").write_text("one\n two \n\none\n", encoding="utf-8")
    (label_dir / "abc_Latn_pbc.txt").write_text("two\nthree\n", encoding="utf-8")
    # Stripped, empty lines dropped, a duplicate kept in the first source by name
    assert read_sentences(label_dir) == [
        ("cbc", "one"),
        ("cbc", "two"),
        ("pbc", "three"),
    ]


def test_split_sentences_small_label():
    sentences = [("src", str(i)) for i in range(50)]
    splits = split_sentences(sentences)
    assert {split: len(rows) for split, rows in splits.items()} == {
        "test": 5,
        "dev": 5,
        "train": 40,
    }
    # Every sentence in exactly one split
    assert sorted(sum(splits.values(), [])) == sorted(sentences)


def test_split_sentences_caps():
    sentences = [("src", str(i)) for i in range(150_000)]
    splits = split_sentences(sentences)
    assert {split: len(rows) for split, rows in splits.items()} == {
        "test": CAPS["test"],
        "dev": CAPS["dev"],
        "train": CAPS["train"],
    }
    assert not set(splits["train"]) & (set(splits["dev"]) | set(splits["test"]))
    assert not set(splits["dev"]) & set(splits["test"])


def test_shuffle_is_reproducible():
    sentences = [("src", str(i)) for i in range(100)]
    first, second = list(sentences), list(sentences)
    random.Random("42-abc_Latn").shuffle(first)
    random.Random("42-abc_Latn").shuffle(second)
    assert split_sentences(first) == split_sentences(second)


def test_write_sentences_one_file_per_source(tmp_path):
    label_dir = tmp_path / "train" / "abc_Latn"
    write_sentences([("cbc", "one"), ("pbc", "two"), ("cbc", "three")], label_dir)
    assert (label_dir / "abc_Latn_cbc.txt").read_text(
        encoding="utf-8"
    ) == "one\nthree\n"
    assert (label_dir / "abc_Latn_pbc.txt").read_text(encoding="utf-8") == "two\n"
