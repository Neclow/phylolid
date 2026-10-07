"""Off-the-shelf language-ID models: where each comes from, and whether its languages count for the known set."""

from typing import Final

from .pretrained import (
    CLD3Model,
    ConLIDModel,
    FasttextModel,
    FunLangIDModel,
    PyfrancModel,
    TransformersModel,
)

# Keys are Hugging Face repository IDs, or <org>/<name> for models published elsewhere.
# The union of their languages defines the known set
TEXT_SET_MODELS: Final = {
    "fasttext/lid.176": {
        "model": FasttextModel,
        "url": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    },
    "facebook/fasttext-language-identification": {
        "model": FasttextModel,
        "filename": "model.bin",
        "revision": "3af127d4124fc58b75666f3594bb5143b9757e78",
    },
    "laurievb/OpenLID": {
        "model": FasttextModel,
        "filename": "model.bin",
        "revision": "be5bd6c7f8d7c89e17d6bf1b37fff377c3f5430e",
    },
    "laurievb/OpenLID-v2": {
        "model": FasttextModel,
        "filename": "model.bin",
        "revision": "10b538ebcc452255917c8bdfc38faa49af134cd2",
    },
    "HPLT/OpenLID-v3": {
        "model": FasttextModel,
        "filename": "openlid-v3.bin",
        "revision": "6b9560483e17e42f48d86cebf22b4b58dffeaa70",
    },
    "google/cld3": {"model": CLD3Model},
    # XLM-V fine-tuned on FLEURS
    "juliensimon/xlm-v-base-language-id": {
        "model": TransformersModel,
        "revision": "43f6bb2b173010e0b982bdd98cff549c754909cf",
    },
    # franc-all as ported by pyfranc: 380 languages in its data (its README says 414)
    "cyb3rk0tik/pyfranc": {"model": PyfrancModel},
}

# Trained on, or covering, most of GlotLID-C, so they cannot define the known set: evaluation only
TEXT_EVAL_MODELS: Final = {
    "cis-lmu/glotlid": {
        "model": FasttextModel,
        "filename": "model_v3.bin",
        "revision": "85cd6716494360367b75f642b5bc78667605d0b4",
    },
    # No licence on the model card or the code repository: local evaluation only, never redistributed
    "epfl-nlp/ConLID": {
        "model": ConLIDModel,
        "revision": "bb370277ea6ebb4f801868db5a4d3dbb3f2e7fd6",
    },
    # google-research/url-nlp, commit of 2025-01-10
    "google-research/fun_langid": {
        "model": FunLangIDModel,
        "url": "https://raw.githubusercontent.com/google-research/url-nlp/ec1cfc8ca7c1e26a59f122f4a2a5bb94910bfa15/fun_langid/fun_langid.py",
    },
}

TEXT_MODELS: Final = {**TEXT_SET_MODELS, **TEXT_EVAL_MODELS}

# As in phylaudio, each model is tagged with its input type, so that audio models can join later
MODEL_ZOO: Final = {
    model_id: {**entry, "dtype": "text"} for model_id, entry in TEXT_MODELS.items()
}
