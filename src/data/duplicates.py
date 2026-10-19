"""Near-duplicate sentences across splits: the same text up to case, punctuation, digits and spacing, or held in one sentence of the other split (all its word 4-grams in it: the contamination rule of GlotLID and AfroScope)."""

from collections import Counter

import regex

# Words per n-gram (GlotLID, AfroScope)
N = 4
# Anything but letters and marks (Unicode categories L and M)
NOT_LETTERS = regex.compile(r"[^\p{L}\p{M}]+")


def normalize(text):
    """Letters and marks only, lowercased: case, punctuation, digits (verse numbers) and spacing removed."""
    return NOT_LETTERS.sub("", text.lower())


def ngrams(text, n=N):
    """The text's word n-grams, words split on whitespace and normalized (hashed, to keep an index small)."""
    words = [word for word in map(normalize, text.split()) if word]
    return {hash(tuple(words[i : i + n])) for i in range(len(words) - n + 1)}


class Index:
    """The sentences of one split (e.g. a class's training sentences), in which to look up sentences of another. Python
    salts hash() per process: build and query an index in the same one."""

    def __init__(self, sentences):
        self.normalized = {normalize(sentence) for sentence in sentences}
        self.postings = {}
        for i, sentence in enumerate(sentences):
            for gram in ngrams(sentence):
                self.postings.setdefault(gram, []).append(i)

    def exact(self, sentence):
        """Whether an indexed sentence is the same text once normalized."""
        return normalize(sentence) in self.normalized

    def match(self, sentence):
        """The indexed sentence holding the largest share of the sentence's word 4-grams: (share, its index); share 1:
        all of them (the contamination rule). (0, None) for a sentence of fewer than 4 words or with none indexed.
        """
        grams = ngrams(sentence)
        counts = Counter()
        for gram in grams:
            counts.update(self.postings.get(gram, ()))
        if not counts:
            return 0.0, None
        i, count = counts.most_common(1)[0]
        return count / len(grams), i
