"""Datasets over the sentence splits of data/splits."""

import os
import random
import unicodedata

from glob import glob

import numpy as np
import torch

from torch.utils.data import Dataset

from src._config import DEFAULT_SPLIT_DIR

# Scripts whose combining marks are optional in writing (diacritics, tone marks, Arabic vowel marks), so that spelling
# augmentation may drop them; in others, such as Devanagari, combining marks are vowels
PLAIN_SCRIPTS = {"Latn", "Arab"}


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


def plain_spelling(text):
    """The text lowercased and without combining marks (diacritics, tone marks, vowel marks), as speech transcripts and
    everyday writing often spell languages whose Bible marks them."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    kept = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", kept)


class FeatureDataset(Dataset):
    """Sentences of one split as fastText feature ids (src/lid/features.py) and class indices. The texts are held as one
    UTF-8 buffer, not Python strings, so that DataLoader workers share it instead of copying it. With plain > 0, that
    share of the sentences in a PLAIN_SCRIPTS script is read in plain_spelling (augmentation, for training).
    """

    def __init__(
        self, split, classes, features, split_dir=DEFAULT_SPLIT_DIR, plain=0.0
    ):
        index = {glottocode: i for i, glottocode in enumerate(classes)}
        texts, targets, plainable = [], [], []
        for path in sorted(glob(f"{split_dir}/{split}/*/*.txt")):
            with open(path, encoding="utf-8") as f:
                # Split on \n only, as TextDataset
                lines = f.read().rstrip("\n").split("\n")
            texts += [line.encode("utf-8") for line in lines]
            targets += [index[os.path.basename(os.path.dirname(path))]] * len(lines)
            # Files <iso>_<Script>_<source>.txt
            script = os.path.basename(path).split("_")[1]
            plainable += [script in PLAIN_SCRIPTS] * len(lines)
        self.buffer = np.frombuffer(b"".join(texts), dtype=np.uint8)
        self.offsets = np.cumsum([0] + [len(text) for text in texts])
        self.targets = np.array(targets)
        self.plainable = np.array(plainable)
        self.plain = plain
        self.features = features

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        start, end = self.offsets[index], self.offsets[index + 1]
        text = self.buffer[start:end].tobytes().decode("utf-8")
        # PyTorch seeds random in each DataLoader worker
        if self.plain and self.plainable[index] and random.random() < self.plain:
            text = plain_spelling(text)
        return self.features(text), self.targets[index]


def collate_features(batch):
    """(feature ids, class) pairs -> EmbeddingBag inputs: all ids in one tensor, where each text starts, the classes."""
    ids = [torch.tensor(text_ids) for text_ids, _ in batch]
    starts = torch.tensor([0] + [len(text_ids) for text_ids in ids[:-1]]).cumsum(0)
    targets = torch.tensor([target for _, target in batch])
    return {"ids": torch.cat(ids), "offsets": starts, "target": targets}
