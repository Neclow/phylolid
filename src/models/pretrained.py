"""Off-the-shelf language-ID models, loaded from their published sources (and downloaded the first time)."""

import os
import runpy

from abc import ABC, abstractmethod

import fasttext
import gcld3
import requests

from huggingface_hub import hf_hub_download, snapshot_download
from pyfranc import franc
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


class CLD3Model(BaseLIDModel):
    """Google's CLD3, built into the gcld3 package: nothing to download."""

    def load(self):
        # Byte limits as in CLD3's README
        self.model = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)


class PyfrancModel(BaseLIDModel):
    """pyfranc, a port of franc-all whose trigram profiles are built into the package: nothing to download."""

    def load(self):
        self.model = franc


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
