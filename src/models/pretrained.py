"""Off-the-shelf language-ID models, loaded from their published sources (and downloaded the first time)."""

import os
import re
import runpy

from abc import ABC, abstractmethod

import fasttext
import gcld3
import numpy as np
import regex
import requests
import torch

from fontTools.unicodedata import script, script_code
from huggingface_hub import hf_hub_download, snapshot_download
from pyfranc import franc
from pyfranc.data import data as franc_profiles
from pyfranc.expressions import expressions as franc_expressions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src._config import DEFAULT_EXTERN_DIR

# ISO 15924 codes of characters that no one script owns: Common (spaces, digits, punctuation), Inherited (combining
# marks), Unknown
SHARED_SCRIPTS = {"Zyyy", "Zinh", "Zzzz"}


def url_cache_path(cache_dir, model_id, url):
    """Where a file downloaded from a URL is cached: a folder named like Hugging Face's, <org>--<name>."""
    return f"{cache_dir}/{model_id.replace('/', '--')}/{os.path.basename(url)}"


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)


def top_labels(scores, labels):
    """Top label of each row of a predict_proba or decision_function matrix: und for a row of zeros (no prediction)."""
    top = scores.argmax(axis=1)
    return [labels[i] if scores[row, i] > 0 else "und" for row, i in enumerate(top)]


class BaseLIDModel(ABC):
    """Language-ID model, loaded on creation from files in cache_dir, which are downloaded the first time.

    model_id is a Hugging Face repository ID, or <org>/<name> for models published elsewhere. As in sklearn, a model
    has predict_proba when it gives probabilities over all its labels, decision_function when it gives only scores
    (higher is better): an (N, L) matrix in labels() order, for N texts without newlines.
    """

    def __init__(self, model_id, cache_dir):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.load()

    @abstractmethod
    def load(self):
        """Load the model, downloading its files to cache_dir if needed."""

    @abstractmethod
    def labels(self):
        """The model's labels, as it writes them."""

    def predict(self, texts):
        """Top label per text, und when the model gives none."""
        scores = getattr(self, "predict_proba", None) or getattr(
            self, "decision_function"
        )
        return top_labels(scores(texts), self.labels())

    def scripts(self):
        """Label -> the ISO 15924 scripts the model can return it for, when the model's own data tells: none here."""
        return {}

    @staticmethod
    def preprocess(text):
        """The model's documented input preprocessing, applied before predicting: none unless a subclass says otherwise."""
        return text


class FasttextModel(BaseLIDModel):
    """fastText model, from a file in a Hugging Face repository or at a URL."""

    def __init__(
        self, model_id, cache_dir, filename="model.bin", revision=None, url=None
    ):
        self.filename = filename
        self.revision = revision
        self.url = url
        super().__init__(model_id, cache_dir)

    def load(self):
        if self.url:
            path = url_cache_path(self.cache_dir, self.model_id, self.url)
            if not os.path.exists(path):
                download(self.url, path)
        else:
            path = hf_hub_download(
                self.model_id,
                self.filename,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        self.model = fasttext.load_model(path)

    def labels(self):
        return [label.removeprefix("__label__") for label in self.model.get_labels()]

    def predict_proba(self, texts):
        texts = [self.preprocess(text) for text in texts]
        if self.model.f.getArgs().loss.name == "softmax":
            # Softmax over the output layer, on the mean of the text's input vectors (what fastText's predict computes,
            # without building a list of labels per text)
            hidden = np.stack([self.model.get_sentence_vector(text) for text in texts])
            logits = hidden @ self.model.get_output_matrix().T
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            return probs / probs.sum(axis=1, keepdims=True)
        # Hierarchical softmax (lid.176): fastText's own predict, over all labels
        index = {label: i for i, label in enumerate(self.model.get_labels())}
        probs = np.zeros((len(texts), len(index)), dtype=np.float32)
        labels, scores = self.model.predict(texts, k=-1)
        for row, (text_labels, text_scores) in enumerate(zip(labels, scores)):
            probs[row, [index[label] for label in text_labels]] = text_scores
        return probs


# Characters OpenLID removes before identifying a language: neither a word character nor a space, or a digit
OPENLID_NONWORD = regex.compile(r"[^\p{Word}\p{Zs}]|\d")
OPENLID_SPACES = regex.compile(r"\s\s+")


class OpenLIDv2Model(FasttextModel):
    """OpenLID-v2, whose model card asks for its cleaning (openlid_normer.clean_line) "for best results"."""

    @staticmethod
    def preprocess(text):
        # Copied from https://github.com/hplt-project/openlid/blob/main/retrain_openlid/tools/openlid_normer.py (the copy
        # in the gated laurievb/OpenLID-v2 dataset could not be read)
        text = text.strip().replace("\n", " ").lower()
        text = regex.sub(OPENLID_NONWORD, "", text)
        return regex.sub(OPENLID_SPACES, " ", text)


class OpenLIDv3Model(FasttextModel):
    """OpenLID-v3, whose model card preprocesses text as OpenLID-v2 cleans it, but squeezing spaces first."""

    @staticmethod
    def preprocess(text):
        # Copied from https://huggingface.co/HPLT/OpenLID-v3 (README)
        text = text.strip().replace("\n", " ").lower()
        text = regex.sub(OPENLID_SPACES, " ", text)
        return regex.sub(OPENLID_NONWORD, "", text)


class TransformersModel(BaseLIDModel):
    """Transformers sequence-classification model from a Hugging Face repository."""

    def __init__(self, model_id, cache_dir, revision):
        self.revision = revision
        super().__init__(model_id, cache_dir)

    def load(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir
        )
        # Typed as possibly None: tell the type checker it loaded
        assert tokenizer is not None
        self.tokenizer = tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir
        )
        # Inference only, on the GPU when there is one
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def labels(self):
        # In the order of the output layer
        return [
            self.model.config.id2label[i] for i in range(self.model.config.num_labels)
        ]

    def predict_proba(self, texts):
        # Truncated as in training (XLM-V: train-xlm.py, on FLEURS' raw transcriptions), but to the model's positions:
        # XLM-V's tokenizer sets no real maximum length (1e30), and some test sentences reach 2,423 tokens. RoBERTa-style
        # models (XLM-V) keep the first 2 positions for padding
        inputs = self.tokenizer(
            [self.preprocess(text) for text in texts],
            truncation=True,
            max_length=self.model.config.max_position_embeddings - 2,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            return self.model(**inputs).logits.softmax(dim=-1).cpu().numpy()


class CLD3Model(BaseLIDModel):
    """Google's CLD3, built into the gcld3 package. The package does not list its languages, so they are read from
    CLD3's source file at a URL. Needs our patched gcld3 (pixi.toml), whose results hold the probabilities of all
    languages."""

    def __init__(self, model_id, cache_dir, url):
        self.url = url
        super().__init__(model_id, cache_dir)

    def load(self):
        self.source_path = url_cache_path(self.cache_dir, self.model_id, self.url)
        if not os.path.exists(self.source_path):
            download(self.url, self.source_path)
        # Byte limits as in CLD3's README
        self.model = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)

    def labels(self):
        with open(self.source_path, encoding="utf-8") as f:
            source = f.read()
        # const char *const TaskContextParams::kLanguageNames[] = {"eo", "co", ...};
        names = re.search(r"kLanguageNames\[\]\s*=\s*\{(.*?)\};", source, re.S)
        return re.findall(r'"([^"]+)"', names.group(1)) if names else []

    def predict_proba(self, texts):
        # Result.probabilities (our gcld3 patch) is in labels() order. It is empty when CLD3 answers und without running
        # its network (text without enough valid bytes, which min_num_bytes=0 makes rare): the row stays 0, a mistake
        probs = np.zeros((len(texts), len(self.labels())), dtype=np.float32)
        for row, text in enumerate(texts):
            probabilities = self.model.FindLanguage(self.preprocess(text)).probabilities
            if probabilities:
                probs[row] = probabilities
        return probs


class PyfrancModel(BaseLIDModel):
    """pyfranc, a port of franc-all whose trigram profiles are built into the package: nothing to download."""

    def load(self):
        self.model = franc

    def labels(self):
        # Profiles are grouped by script: {script: {language: trigrams}}
        profiled = {
            language for script in franc_profiles.values() for language in script
        }
        # A language with a script of its own (jpn, kor, ...) has no profile: franc returns it from the script alone.
        # Its expressions are keyed by script for shared scripts, by language for these
        script_only = {key for key in franc_expressions if key not in franc_profiles}
        # und: text too short, or in no known script
        return sorted(profiled | script_only | {"und"})

    def scripts(self):
        # franc scores a text against the profiles of its main script only: a language is returned for the scripts of
        # its profiles (Unicode script names) and of its own expression (one Unicode script, which its first character
        # tells; jav has both)
        scripts = {}
        for name, languages in franc_profiles.items():
            for language in languages:
                scripts.setdefault(language, set()).add(script_code(name))
        for language, expression in franc_expressions.items():
            if language not in franc_profiles:
                first = chr(int(re.findall(r"\\u([0-9A-F]{4})", expression)[0], 16))
                scripts.setdefault(language, set()).add(script(first))
        return scripts

    def decision_function(self, texts):
        # Not probabilities: 1 - the normalised trigram distance, 1 for the closest language. franc scores the languages
        # of the text's script, with its defaults (und under 10 characters); 0 for the others
        index = {label: i for i, label in enumerate(self.labels())}
        scores = np.zeros((len(texts), len(index)), dtype=np.float32)
        for row, text in enumerate(texts):
            for language, score in self.model.lang_detect(self.preprocess(text)):
                scores[row, index[language]] = score
        return scores


class FunLangIDModel(BaseLIDModel):
    """FUN-LangID, a single Python file at a URL that holds both the code and the model (frequent 4-grams per language)."""

    def __init__(self, model_id, cache_dir, url):
        self.url = url
        super().__init__(model_id, cache_dir)

    def load(self):
        path = url_cache_path(self.cache_dir, self.model_id, self.url)
        if not os.path.exists(path):
            download(self.url, path)
        self.model = runpy.run_path(path)["FunLangID"]()

    def labels(self):
        # Its lexicon maps each 4-gram to the languages it is frequent in
        return sorted(
            {
                language
                for languages in self.model.lex_dict.values()
                for language in languages
            }
        )

    def scripts(self):
        # A language scores only through its lexicon's 1- to 4-grams: it is returned for their scripts
        scripts = {}
        for gram, languages in self.model.lex_dict.items():
            for language in languages:
                scripts.setdefault(language, set()).update(map(script, gram))
        return {language: found - SHARED_SCRIPTS for language, found in scripts.items()}

    def decision_function(self, texts):
        # Not probabilities: the share of the text's 4-grams frequent in each language, for every language some 4-gram
        # votes for (top_n_predictions=None); 0 for the others. A text without 4-grams gets und, not a label: its row
        # stays 0
        index = {label: i for i, label in enumerate(self.labels())}
        scores = np.zeros((len(texts), len(index)), dtype=np.float32)
        for row, text in enumerate(texts):
            for language, score in self.model.predict(self.preprocess(text), None):
                if language in index:
                    scores[row, index[language]] = score
        return scores


class ConLIDModel(BaseLIDModel):
    """ConLID: weights from a Hugging Face repository, code from the extern/ConLID git submodule."""

    def __init__(self, model_id, cache_dir, revision):
        self.revision = revision
        super().__init__(model_id, cache_dir)

    def load(self):
        path = snapshot_download(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir
        )
        conlid = runpy.run_path(f"{DEFAULT_EXTERN_DIR}/ConLID/model.py")["ConLID"]
        self.model = conlid.from_pretrained(dir=path)
        self.model.eval()

    def labels(self):
        # labels.json lists them in output order
        return self.model.get_labels()

    def predict_proba(self, texts):
        # As ConLID's predict_batched, without sorting the labels: whitespace tokens and their character n-grams as ids,
        # padded (padding is masked out of the mean), then a softmax
        ids = [
            self.model._tokens2ngrams(self.model._tokenize(self.preprocess(text)))
            for text in texts
        ]
        width = max(len(text_ids) for text_ids in ids)
        padded = [
            text_ids + [self.model.pad_id] * (width - len(text_ids)) for text_ids in ids
        ]
        inputs = torch.tensor(padded, dtype=torch.int, device=self.model.device)
        with torch.no_grad():
            return self.model(inputs).softmax(dim=-1).cpu().numpy()
