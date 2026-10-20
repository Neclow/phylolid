"""Unit tests for the text features in src/lid/features.py."""

import fasttext

from src.lid.features import (
    BUCKET,
    MAXN,
    MINN,
    Features,
    normalize_spelling,
    spelled_words,
    tokenize,
    vocab_file,
)


def test_tokenize_matches_fasttext():
    # fastText's separators split words; a no-break space and a Unicode line separator do not
    text = " a\tb\u00a0c\u2028d\x00e\x0bf\x0cg  h\r"
    assert (
        tokenize(text)
        == fasttext.tokenize(text)
        == ["a", "b\u00a0c\u2028d", "e", "f", "g", "h"]
    )


def test_tokenize_without_end_of_line():
    # fastText turns a newline into its end-of-line token </s>; we drop it, as ConLID
    assert fasttext.tokenize("x\ny") == ["x", "</s>", "y"]
    assert tokenize("x\ny\n") == ["x", "y"]


def test_word_ids_match_fasttext(tmp_path):
    train = tmp_path / "train.txt"
    train.write_text(
        "__label__a the cat sat\n__label__b le chat\n__label__a the dog\n",
        encoding="utf-8",
    )
    trained = fasttext.train_supervised(
        str(train),
        dim=4,
        epoch=1,
        minCount=1,
        minn=MINN,
        maxn=MAXN,
        bucket=BUCKET,
        thread=1,
        verbose=0,
    )
    # Same vocabulary in the same order (fastText's includes its end-of-line token)
    features = Features(trained.get_words())
    # Words in the vocabulary and not, with multi-byte characters (accents, CJK, emoji)
    for word in ["the", "chat", "épée", "日本語", "x", "😀ab"]:
        _, ids = trained.get_subwords(word)
        assert features.word_ids(word) == ids.tolist(), word
    assert features("the épée") == features.word_ids("the") + features.word_ids("épée")


def test_normalize_spelling():
    # Case, Latin diacritics and tone marks, Arabic vowel marks go; letters without a decomposition stay
    assert normalize_spelling("Kɛ̄ Jɛhova Ɓàá ŋ̀ ø Épée") == "kɛ jɛhova ɓaa ŋ ø epee"
    assert normalize_spelling("كَتَبَ") == "كتب"
    # Devanagari's combining marks are vowels, and Greek is left as it is
    assert normalize_spelling("हिंदी") == "हिंदी"
    assert normalize_spelling("Ελληνικά") == "ελληνικά"


def test_spelled_words():
    assert spelled_words("Le château de", "raw") == ["Le", "château", "de"]
    assert spelled_words("Le château de", "normalized") == ["le", "chateau", "de"]
    # A word normalization leaves as it is counts once
    assert spelled_words("Le château de", "both") == [
        "Le",
        "le",
        "château",
        "chateau",
        "de",
    ]


def test_features_both_spellings():
    features = Features(["chateau"], spelling="both")
    raw, normalized = Features(["chateau"]), Features(
        ["chateau"], spelling="normalized"
    )
    # The ids of the word as written, then of its normalized form (a vocabulary word)
    assert features("château") == raw("château") + normalized("château")
    assert normalized("château")[0] == 0
    assert vocab_file("raw").endswith("/vocab.json")
    assert vocab_file("both").endswith("/vocab_both.json")
