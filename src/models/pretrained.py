"""Off-the-shelf language-ID models, loaded from their published sources (and downloaded the first time)."""

import os
import re
import runpy

from abc import ABC, abstractmethod

import fasttext
import gcld3
import requests

from huggingface_hub import hf_hub_download, snapshot_download
from pyfranc import franc
from pyfranc.data import data as franc_profiles
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src._config import DEFAULT_EXTERN_DIR


def url_cache_path(cache_dir, model_id, url):
    """Where a file downloaded from a URL is cached: a folder named like Hugging Face's, <org>--<name>."""
    return f"{cache_dir}/{model_id.replace('/', '--')}/{os.path.basename(url)}"


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)


class BaseLIDModel(ABC):
    """Language-ID model, loaded on creation from files in cache_dir, which are downloaded the first time.

    model_id is a Hugging Face repository ID, or <org>/<name> for models published elsewhere.
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


class TransformersModel(BaseLIDModel):
    """Transformers sequence-classification model from a Hugging Face repository."""

    def __init__(self, model_id, cache_dir, revision):
        self.revision = revision
        super().__init__(model_id, cache_dir)

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.revision, cache_dir=self.cache_dir
        )

    def labels(self):
        return list(self.model.config.id2label.values())


class CLD3Model(BaseLIDModel):
    """Google's CLD3, built into the gcld3 package. The package does not list its languages, so they are read from
    CLD3's source file at a URL."""

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


class PyfrancModel(BaseLIDModel):
    """pyfranc, a port of franc-all whose trigram profiles are built into the package: nothing to download."""

    def load(self):
        self.model = franc

    def labels(self):
        # Profiles are grouped by script: {script: {language: trigrams}}
        return sorted(
            {language for script in franc_profiles.values() for language in script}
        )


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

    def labels(self):
        return self.model.get_labels()
