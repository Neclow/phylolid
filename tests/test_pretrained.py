"""Unit tests for the off-the-shelf model wrappers in src/models/pretrained.py."""

import fasttext
import numpy as np
import pytest

from src.models.pretrained import FasttextModel, OpenLIDv2Model, OpenLIDv3Model

TRAIN = {
    "eng": ["the cat sat on the mat", "a dog and a cat", "the house is big"],
    "fra": ["le chat est sur le tapis", "un chien et un chat", "la maison est grande"],
    "deu": ["die katze sitzt", "ein hund und eine katze", "das haus ist gross"],
}


def test_openlid_v3_preprocess_card_example():
    # The model card's example sentence: lowercased, final full stop removed
    text = "Maskinsjefen er oppteken av å løfta fram dei maritime utdanningane."
    assert (
        OpenLIDv3Model.preprocess(text)
        == "maskinsjefen er oppteken av å løfta fram dei maritime utdanningane"
    )


def test_openlid_cleaning_order():
    # v2 removes punctuation and digits, then squeezes spaces; v3 squeezes first, so a removed comma leaves two spaces
    text = " Hello , World 42! "
    assert OpenLIDv2Model.preprocess(text) == "hello world "
    assert OpenLIDv3Model.preprocess(text) == "hello  world "
    # Other fastText models take the text as it is
    assert FasttextModel.preprocess(text) == text


@pytest.mark.parametrize("loss", ["softmax", "hs"])
def test_fasttext_predict_matches_fasttext(tmp_path, loss):
    train = tmp_path / "train.txt"
    train.write_text(
        "".join(
            f"__label__{label} {line}\n"
            for label, lines in TRAIN.items()
            for line in lines
        ),
        encoding="utf-8",
    )
    trained = fasttext.train_supervised(
        str(train),
        loss=loss,
        dim=8,
        epoch=20,
        minCount=1,
        minn=2,
        maxn=3,
        bucket=1000,
        thread=1,
        verbose=0,
    )
    # Saved where FasttextModel caches a file downloaded from a URL, so it loads without downloading
    (tmp_path / "test--tiny").mkdir()
    trained.save_model(str(tmp_path / "test--tiny" / "tiny.bin"))
    model = FasttextModel(
        "test/tiny", str(tmp_path), url="https://example.org/tiny.bin"
    )
    texts = ["the cat", "le chien est grand", "ein haus", "x"]
    probs = model.predict(texts)
    assert probs.shape == (len(texts), len(TRAIN))
    assert probs.sum(axis=1) == pytest.approx(np.ones(len(texts)), abs=1e-4)
    # Same probabilities as fastText's own predict over all labels
    labels, scores = model.model.predict(texts, k=-1)
    names = model.labels()
    for row, (text_labels, text_scores) in enumerate(zip(labels, scores)):
        for label, score in zip(text_labels, text_scores):
            assert probs[
                row, names.index(label.removeprefix("__label__"))
            ] == pytest.approx(score, abs=1e-4)
