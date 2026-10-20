"""Unit tests for the datasets in src/data/datasets.py."""

from torch.utils.data import DataLoader

from src.data.datasets import (
    FeatureDataset,
    TextDataset,
    collate_features,
    plain_spelling,
)
from src.lid.features import Features


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_text_dataset(tmp_path):
    write(tmp_path / "test" / "abcd1234" / "abc_Latn_cbc.txt", "one\ntwo\n")
    write(tmp_path / "test" / "abcd1234" / "abc_Latn_omnilingual.txt", "three\n")
    # A Unicode line separator inside a sentence stays in it
    write(tmp_path / "test" / "efgh1234" / "efg_Latn_pbc.txt", "four\u2028four\n")
    write(tmp_path / "heldout" / "ijkl1234" / "ijk_Latn_jw.txt", "five\n")
    dataset = TextDataset("test", split_dir=tmp_path)
    assert len(dataset) == 4
    assert [item["text"] for item in dataset] == [
        "one",
        "two",
        "three",
        "four\u2028four",
    ]
    assert dataset[2] == {
        "text": "three",
        "glottocode": "abcd1234",
        "file": "abc_Latn_omnilingual",
        "dataset": "omnilingual-asr",
    }
    assert dataset[3]["dataset"] == "glotlid-corpus"


def test_text_dataset_batches(tmp_path):
    write(tmp_path / "dev" / "abcd1234" / "abc_Latn_cbc.txt", "one\ntwo\nthree\n")
    batch = next(iter(DataLoader(TextDataset("dev", split_dir=tmp_path), batch_size=2)))
    # The default collate turns a batch of dicts into a dict of lists
    assert batch["text"] == ["one", "two"]
    assert batch["glottocode"] == ["abcd1234", "abcd1234"]


def test_feature_dataset_and_batches(tmp_path):
    write(tmp_path / "train" / "abcd1234" / "abc_Latn_cbc.txt", "the cat\nle chat\n")
    write(tmp_path / "train" / "efgh1234" / "efg_Latn_pbc.txt", "épée\n")
    features = Features(["the"])
    dataset = FeatureDataset(
        "train", ["efgh1234", "abcd1234"], features, split_dir=tmp_path
    )
    assert len(dataset) == 3
    # Texts read back from the shared buffer, multi-byte characters included; classes in the given order
    assert dataset[0] == (features("the cat"), 1)
    assert dataset[2] == (features("épée"), 0)
    batch = collate_features([dataset[0], dataset[2]])
    assert batch["ids"].tolist() == features("the cat") + features("épée")
    assert batch["offsets"].tolist() == [0, len(features("the cat"))]
    assert batch["target"].tolist() == [1, 0]


def test_plain_spelling():
    # Case, diacritics and tone marks go (Latin), as Arabic vowel marks; letters without a decomposition stay
    assert plain_spelling("Kɛ̄ Jɛhova Ɓàá ŋ̀ ø") == "kɛ jɛhova ɓaa ŋ ø"
    assert plain_spelling("كَتَبَ") == "كتب"


def test_feature_dataset_plain_spelling(tmp_path):
    write(tmp_path / "train" / "abcd1234" / "abc_Latn_cbc.txt", "Épée\n")
    write(tmp_path / "train" / "efgh1234" / "efg_Deva_pbc.txt", "हिंदी\n")
    features = Features([])
    dataset = FeatureDataset(
        "train", ["abcd1234", "efgh1234"], features, split_dir=tmp_path, plain=1.0
    )
    # Latin script in plain spelling; Devanagari kept, as its combining marks are vowels
    assert dataset[0] == (features("epee"), 0)
    assert dataset[1] == (features("हिंदी"), 1)
