"""fastText's text features for the language-ID models we train: words and hashed character n-grams."""

import json
import re
import unicodedata

from functools import lru_cache

import regex

from src._config import DEFAULT_SPLIT_DIR

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
# Combining marks on a Latin or Arabic letter: diacritics, tone marks and Arabic vowel marks, often left out in
# writing. In other scripts, such as Devanagari, combining marks are vowels, and stay
OPTIONAL_MARKS = regex.compile(r"(?<=[\p{Latin}\p{Arabic}]\p{M}*)\p{M}")


# How the features read a text's spelling: as written; normalized (normalize_spelling); or both, each word that
# normalization changes also counting in its normalized form
SPELLINGS = ["raw", "normalized", "both"]


def normalize_spelling(text):
    """The text lowercased and without the combining marks of Latin and Arabic letters, as speech transcripts and
    everyday writing often spell languages whose Bible marks them."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return unicodedata.normalize("NFC", OPTIONAL_MARKS.sub("", decomposed))


def spelled_words(text, spelling):
    """The words of a text in one of SPELLINGS: "Le château" -> raw Le château; normalized le chateau; both Le le
    château chateau."""
    if spelling == "normalized":
        return tokenize(normalize_spelling(text))
    words = tokenize(text)
    if spelling == "both":
        words = [
            form
            for word in words
            for form in dict.fromkeys([word, normalize_spelling(word)])
        ]
    return words


def vocab_file(spelling):
    """The vocabulary of a spelling, written by pipeline/build_vocab.py."""
    suffix = "" if spelling == "raw" else f"_{spelling}"
    return f"{DEFAULT_SPLIT_DIR}/vocab{suffix}.json"


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
    come after the vocabulary's (Dictionary::addSubwords). The words are read in one of SPELLINGS (spelled_words).
    """

    def __init__(self, vocab, spelling="raw"):
        # The vocabulary's words, most frequent first: a word's position is its id
        self.index = {word: i for i, word in enumerate(vocab)}
        self.size = len(self.index) + BUCKET
        self.spelling = spelling
        self.word_ids = lru_cache(maxsize=CACHE_SIZE)(self._word_ids)

    @classmethod
    def from_json(cls, path, spelling="raw"):
        """Features over a vocabulary written by pipeline/build_vocab.py."""
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f), spelling)

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
        words = spelled_words(text, self.spelling)
        return [i for word in words for i in self.word_ids(word)]
