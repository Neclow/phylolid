"""Unit tests for the datasets in src/data/datasets.py."""

from torch.utils.data import DataLoader

from src.data.datasets import TextDataset


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
