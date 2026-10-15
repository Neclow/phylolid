"""Unit tests for the off-the-shelf model wrappers in src/models/pretrained.py."""

import runpy

import fasttext
import numpy as np
import pytest

from src._config import DEFAULT_EXTERN_DIR
from src.models.pretrained import (
    CLD3Model,
    ConLIDModel,
    FasttextModel,
    FunLangIDModel,
    OpenLIDv2Model,
    OpenLIDv3Model,
    PyfrancModel,
    top_labels,
)

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
    probs = model.predict_proba(texts)
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


def test_cld3_predict_all_languages(tmp_path):
    # CLD3's own label list (kLanguageNames at commit b48dc46, its output order), where the wrapper caches it
    labels = """eo co eu ta de mt ps te su uz zh-Latn ne nl sw sq hmn ja no mn so ko kk sl ig mr th zu ml hr bs lo sd cy
    hy uk pt lv iw cs vi jv be km mk tr fy am zh da sv fi ht af la id fil sm ca el ka sr it sk ru ru-Latn bg ny fa haw gl
    et ms gd bg-Latn ha is ur mi hi bn hi-Latn fr yi hu xh my tg ro ar lb el-Latn st ceb kn az si ky mg en gu es pl ja-Latn
    ga lt sn yo pa ku""".split()
    (tmp_path / "google--cld3").mkdir()
    names = ", ".join(f'"{label}"' for label in labels)
    (tmp_path / "google--cld3" / "task_context_params.cc").write_text(
        f"const char *const TaskContextParams::kLanguageNames[] = {{{names}, nullptr}};",
        encoding="utf-8",
    )
    model = CLD3Model(
        "google/cld3", str(tmp_path), url="https://example.org/task_context_params.cc"
    )
    assert hasattr(model, "predict_proba")
    texts = [
        "Ceci est une phrase en français, assez longue pour CLD3.",
        "This is an English sentence.",
    ]
    probs = model.predict_proba(texts)
    assert probs.shape == (2, len(labels))
    assert probs.sum(axis=1) == pytest.approx([1, 1], abs=1e-4)
    # Same top language and probability as CLD3's FindLanguage
    for text, row in zip(texts, probs):
        result = model.model.FindLanguage(text)
        assert labels[row.argmax()] == result.language
        assert row.max() == pytest.approx(result.probability)


def test_conlid_predict_matches_conlid():
    # A tiny ConLID with random weights: two words, 50 n-gram buckets, two labels
    conlid = runpy.run_path(f"{DEFAULT_EXTERN_DIR}/ConLID/model.py")["ConLID"]
    vocab = {"<pad>": 0, "<unk>": 1, "the": 2, "le": 3}
    tiny = conlid(
        len(vocab) + 50, 8, 2, bucket=50, min_count=1, minn=2, maxn=3, aggr="mean"
    )
    tiny.vocab, tiny.vocab_size = vocab, len(vocab)
    tiny.id2label = {0: "eng_Latn", 1: "fra_Latn"}
    tiny.to(tiny.device).eval()
    # The wrapper around it, without downloading ConLID's weights
    model = ConLIDModel.__new__(ConLIDModel)
    model.model = tiny
    texts = ["the cat", "le chat est sur le tapis", "x"]
    probs = model.predict_proba(texts)
    assert probs.shape == (3, 2)
    # Same probabilities as ConLID's own predict_batched over all labels
    labels, scores = tiny.predict_batched(texts, k=-1)
    for row, (text_labels, text_scores) in enumerate(zip(labels, scores)):
        for label, score in zip(text_labels, text_scores):
            assert probs[row, model.labels().index(label)] == pytest.approx(
                score, abs=1e-6
            )


def test_pyfranc_predict_scores():
    model = PyfrancModel("cyb3rk0tik/pyfranc", "unused")
    labels = model.labels()
    texts = [
        "Ceci est une phrase en français, assez longue.",
        "short",
        "これは日本語の文章です。",
    ]
    scores = model.decision_function(texts)
    assert not hasattr(model, "predict_proba")
    assert scores.shape == (3, len(labels))
    # The closest language scores 1; under 10 characters, und; a script of its own, its language alone
    assert scores[0].max() == 1 and (scores[0] > 0).sum() > 100
    assert labels[scores[1].argmax()] == "und" and (scores[1] > 0).sum() == 1
    assert labels[scores[2].argmax()] == "jpn" and (scores[2] > 0).sum() == 1


def test_pyfranc_scripts():
    scripts = PyfrancModel("cyb3rk0tik/pyfranc", "unused").scripts()
    # Profiles in two scripts, a script of its own, both, a Unicode script name (Canadian_Aboriginal)
    assert scripts["uig"] == {"Arab", "Latn"}
    assert scripts["blt"] == {"Tavt"}
    assert scripts["jav"] == {"Java", "Latn"}
    assert scripts["ike"] == {"Cans"}
    # franc's Japanese is kana
    assert scripts["jpn"] == {"Hira"}
    assert "und" not in scripts


def test_fun_langid_scripts(tmp_path):
    (tmp_path / "google-research--fun_langid").mkdir()
    (tmp_path / "google-research--fun_langid" / "fun_langid.py").write_text(
        """
class FunLangID:
    lex_dict = {"abcd": ("aaa",), "где": ("bbb",), ", ka": ("aaa", "bbb"), "日": ("ccc",)}
""",
        encoding="utf-8",
    )
    model = FunLangIDModel(
        "google-research/fun_langid",
        str(tmp_path),
        url="https://example.org/fun_langid.py",
    )
    # Scripts of its n-grams' characters; punctuation and spaces belong to none
    assert model.scripts() == {
        "aaa": {"Latn"},
        "bbb": {"Cyrl", "Latn"},
        "ccc": {"Hani"},
    }


def test_fun_langid_predict_scores(tmp_path):
    # A stand-in for fun_langid.py with the same interface, where the wrapper caches it: nothing is downloaded
    (tmp_path / "google-research--fun_langid").mkdir()
    (tmp_path / "google-research--fun_langid" / "fun_langid.py").write_text(
        """
class FunLangID:
    lex_dict = {"abcd": ("aaa",), "bcde": ("aaa", "bbb"), "cdef": ("ccc",)}

    def predict(self, text, top_n_predictions=5):
        return [("und", 0)] if not text else [("aaa", 0.5), ("bbb", 0.25)]
""",
        encoding="utf-8",
    )
    model = FunLangIDModel(
        "google-research/fun_langid",
        str(tmp_path),
        url="https://example.org/fun_langid.py",
    )
    assert model.labels() == ["aaa", "bbb", "ccc"]
    assert not hasattr(model, "predict_proba")
    scores = model.decision_function(["some text", ""])
    assert scores[0].tolist() == [0.5, 0.25, 0]
    # und is not a label: the row stays 0
    assert scores[1].tolist() == [0, 0, 0]


def test_top_labels_und_for_rows_of_zeros():
    scores = np.array([[0.2, 0.7, 0.1], [0, 0, 0]])
    assert top_labels(scores, ["a", "b", "c"]) == ["b", "und"]


def test_predict_top_label():
    # Through decision_function for pyfranc, which has no predict_proba
    model = PyfrancModel("cyb3rk0tik/pyfranc", "unused")
    assert model.predict(["short", "これは日本語の文章です。"]) == ["und", "jpn"]
