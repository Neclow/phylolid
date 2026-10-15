"""Unit tests for the text features in src/lid/features.py."""

import fasttext

from src.lid.features import BUCKET, MAXN, MINN, Features, tokenize


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
