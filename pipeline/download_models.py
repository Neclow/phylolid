"""Download the off-the-shelf language-ID models of the model zoo."""

from src._config import DEFAULT_CACHE_DIR
from src.models._model_zoo import MODEL_ZOO, load_model


def main():
    for model_id in MODEL_ZOO:
        print(f"Downloading model: {model_id}")
        # Loading a model downloads its files the first time
        model = load_model(model_id, DEFAULT_CACHE_DIR)
        del model


if __name__ == "__main__":
    main()
