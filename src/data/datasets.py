"""Datasets over the sentence splits of data/splits."""

from pathlib import Path

from torch.utils.data import Dataset

from src._config import DEFAULT_SPLIT_DIR


class TextDataset(Dataset):
    """Sentences of one split (train, dev, test or heldout): text, class glottocode, file and dataset."""

    def __init__(self, split, split_dir=DEFAULT_SPLIT_DIR):
        self.items = []
        # Files <split>/<glottocode>/<label>_<source>.txt; Omnilingual sentences are <label>_omnilingual.txt
        for path in sorted(Path(split_dir, split).glob("*/*.txt")):
            dataset = (
                "omnilingual-asr"
                if path.stem.endswith("_omnilingual")
                else "glotlid-corpus"
            )
            # Split on \n only: a few sentences keep Unicode line separators, which splitlines() would cut
            for text in path.read_text(encoding="utf-8").rstrip("\n").split("\n"):
                self.items.append(
                    {
                        "text": text,
                        "glottocode": path.parent.name,
                        "file": path.stem,
                        "dataset": dataset,
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
