"""Datasets over the sentence splits of data/splits."""

import os

from glob import glob

from torch.utils.data import Dataset

from src._config import DEFAULT_SPLIT_DIR


class TextDataset(Dataset):
    """Sentences of one split (train, dev, test or heldout): text, class glottocode, file and dataset."""

    def __init__(self, split, split_dir=DEFAULT_SPLIT_DIR):
        self.items = []
        # Files <split>/<glottocode>/<label>_<source>.txt; Omnilingual sentences are <label>_omnilingual.txt
        for path in sorted(glob(f"{split_dir}/{split}/*/*.txt")):
            file = os.path.basename(path).removesuffix(".txt")
            dataset = (
                "omnilingual-asr" if file.endswith("_omnilingual") else "glotlid-corpus"
            )
            with open(path, encoding="utf-8") as f:
                # Split on \n only: a few sentences keep Unicode line separators, which splitlines() would cut
                texts = f.read().rstrip("\n").split("\n")
            for text in texts:
                self.items.append(
                    {
                        "text": text,
                        "glottocode": os.path.basename(os.path.dirname(path)),
                        "file": file,
                        "dataset": dataset,
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
