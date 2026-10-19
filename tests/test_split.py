"""Unit tests for the sentence split in src/data/split.py."""

import random

from src.data.split import (
    CAPS,
    read_lines,
    split_sentences,
    unique_sentences,
    write_sentences,
)


def test_unique_sentences_across_files(tmp_path):
    # Two labels of one class (two scripts), each with a source file
    (tmp_path / "abc_Arab_cbc.txt").write_text("one\n two \n\none\n", encoding="utf-8")
    (tmp_path / "abc_Latn_pbc.txt").write_text("two\nthree\n", encoding="utf-8")
    paths = sorted(tmp_path.glob("*.txt"))
    # Stripped, empty lines dropped, a duplicate kept under the first file
    assert unique_sentences(read_lines(paths)) == [
        ("abc_Arab_cbc", "one"),
        ("abc_Arab_cbc", "two"),
        ("abc_Latn_pbc", "three"),
    ]


def test_unique_sentences_normalized():
    # The same verse with another case, punctuation, verse number or spacing is a duplicate; other words are not
    pairs = [
        ("abc_Latn_pbc", "In the beginning, God created."),
        ("abc_Latn_cbc", "1:1 in the beginning God  created"),
        ("abc_Latn_cbc", "In the beginning God made."),
    ]
    assert unique_sentences(pairs) == [pairs[0], pairs[2]]


def test_split_sentences_small_class():
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


def test_write_sentences_one_file_per_name(tmp_path):
    class_dir = tmp_path / "train" / "abcd1234"
    pairs = [
        ("abc_Latn_cbc", "one"),
        ("abc_Latn_omnilingual", "two"),
        ("abc_Latn_cbc", "three"),
    ]
    write_sentences(pairs, class_dir)
    assert (class_dir / "abc_Latn_cbc.txt").read_text(
        encoding="utf-8"
    ) == "one\nthree\n"
    assert (class_dir / "abc_Latn_omnilingual.txt").read_text(
        encoding="utf-8"
    ) == "two\n"
