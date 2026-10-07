"""Download the off-the-shelf language-ID models of the model zoo."""

from src._config import DEFAULT_CACHE_DIR
from src.models._model_zoo import MODEL_ZOO


def main():
    for model_id, entry in MODEL_ZOO.items():
        print(f"Downloading model: {model_id}")
        # Creating a model loads it, and downloads its files the first time
        model_class = entry["model"]
        # Entries mix classes and strings: tell the type checker this one is a class
        assert isinstance(model_class, type)
        source = {
            key: value for key, value in entry.items() if key not in ("model", "dtype")
        }
        model = model_class(model_id=model_id, cache_dir=DEFAULT_CACHE_DIR, **source)
        del model


if __name__ == "__main__":
    main()
