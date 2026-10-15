"""fastText's text features for the language-ID models we train: words and hashed character n-grams."""

import json
import re

from functools import lru_cache

# fastText's word separators (Dictionary::readWord): space, newline, carriage return, tab, vertical tab, form feed and
# NUL. Other spaces (e.g. no-break space, U+2028) stay inside words
SEPARATORS = re.compile(r"[ \n\r\t\v\f\0]+")
# Character n-grams of <word>, from MINN to MAXN characters, hashed into BUCKET ids: fastText's -minn, -maxn and
# -bucket, as NLLB, OpenLID, GlotLID and ConLID set them
MINN = 2
MAXN = 5
BUCKET = 1_000_000
# Words whose ids are cached, per process: most words of a corpus are frequent ones
CACHE_SIZE = 1 << 20


def tokenize(text):
    """Words of a text, split as fastText splits them, without its end-of-line token (as ConLID)."""
    return [word for word in SEPARATORS.split(text) if word]


def fnv1a(ngram):
    """fastText's hash of a string (Dictionary::hash): 32-bit FNV-1a over its UTF-8 bytes, each read as a signed char."""
    h = 2166136261
    for byte in ngram.encode("utf-8"):
        h = ((h ^ (byte if byte < 128 else byte | 0xFFFFFF00)) * 16777619) & 0xFFFFFFFF
    return h


class Features:
    """fastText's feature ids of a text: each word's id in the vocabulary (if it has one), then its n-grams' ids, which
    come after the vocabulary's (Dictionary::addSubwords)."""

    def __init__(self, vocab):
        # The vocabulary's words, most frequent first: a word's position is its id
        self.index = {word: i for i, word in enumerate(vocab)}
        self.size = len(self.index) + BUCKET
        self.word_ids = lru_cache(maxsize=CACHE_SIZE)(self._word_ids)

    @classmethod
    def from_json(cls, path):
        """Features over a vocabulary written by pipeline/build_vocab.py."""
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def _word_ids(self, word):
        token = f"<{word}>"
        # N-grams by start, then length, as Dictionary::computeSubwords; Python slices count characters, not bytes
        ngrams = [
            token[i : i + n]
            for i in range(len(token))
            for n in range(MINN, MAXN + 1)
            if i + n <= len(token)
        ]
        ids = [self.index[word]] if word in self.index else []
        return ids + [len(self.index) + fnv1a(ngram) % BUCKET for ngram in ngrams]

    def __call__(self, text):
        return [i for word in tokenize(text) for i in self.word_ids(word)]
