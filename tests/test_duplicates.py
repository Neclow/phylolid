"""Unit tests for the near-duplicate checks in src/data/duplicates.py."""

from src.data.duplicates import Index, normalize


def test_normalize():
    # A verse number, case, punctuation and spacing go; letters with diacritics stay
    assert normalize("1:2 Au commencement,  Dieu créa…") == "aucommencementdieucréa"


def test_exact():
    index = Index(["Au commencement, Dieu créa les cieux et la terre."])
    assert index.exact("1 au commencement Dieu créa les cieux et la terre")
    assert not index.exact("Au commencement Dieu créa les cieux.")


def test_match():
    index = Index(["a b c d e f", "c d e f g h"])
    # All 4-grams in the first sentence: held in it (the contamination rule)
    assert index.match("b c d e") == (1.0, 0)
    # One of two 4-grams: half
    assert index.match("A, b c d x") == (0.5, 0)
    # 4-grams spread over two sentences: the second holds 3 of 4, the first 2
    assert index.match("b c d e f g h") == (0.75, 1)
    # Fewer than four words, or nothing indexed
    assert index.match("a b c") == (0.0, None)
    assert index.match("w x y z") == (0.0, None)
