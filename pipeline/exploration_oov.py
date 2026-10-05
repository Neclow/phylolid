"""Run fastText language-ID models on labelled sentences and save their full prediction vectors.

Each input is a TSV with columns `ground_truth` (e.g. kal_Latn) and `text`; other columns are ignored.
For each input and model, writes OUTDIR/<input name>/<model>.csv: one row per sentence, in input order,
with the ground truth and the model's probability for every label.

Example:
    pixi run python drafts.py data/samples/glotlid.tsv
"""
import argparse
import csv
from pathlib import Path

import fasttext
import pandas as pd
import regex
from huggingface_hub import hf_hub_download

# OpenLID preprocessing, from https://huggingface.co/HPLT/OpenLID-v3
NONWORD_REPLACE_PATTERN = regex.compile(r"[^\p{Word}\p{Zs}]|\d")  # either (not a word nor a space) or (is digit)
SPACE_PATTERN = regex.compile(r"\s\s+")  # squeezes sequential whitespace


def openlid_preprocess(text):
    text = text.strip().replace("\n", " ").lower()
    text = regex.sub(SPACE_PATTERN, " ", text)
    text = regex.sub(NONWORD_REPLACE_PATTERN, "", text)
    return text


def single_line(text):
    # The NLLB model card feeds raw text; fastText only requires it to be on one line
    return text.strip().replace("\n", " ")


# name -> (HF repo, file, preprocessing)
MODELS = {
    "openlid-v3": ("HPLT/OpenLID-v3", "openlid-v3.bin", openlid_preprocess),
    "nllb-lid218": ("facebook/fasttext-language-identification", "model.bin", single_line),
}


def predict(model, texts):
    """Probability of every label for each text: one row per text, one column per label."""
    labels = [label.removeprefix("__label__") for label in model.get_labels()]
    rows = []
    for text in texts:
        predicted, probs = model.predict(text, k=-1, on_unicode_error="strict")
        rows.append(dict(zip((label.removeprefix("__label__") for label in predicted), probs)))
    return pd.DataFrame(rows, columns=labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="TSV file(s) with `ground_truth` and `text` columns")
    parser.add_argument("--outdir", type=Path, default=Path("data/predictions"), help="default: %(default)s")
    args = parser.parse_args()

    models = {
        name: (fasttext.load_model(hf_hub_download(repo_id=repo, filename=file, cache_dir="data/models")), preprocess)
        for name, (repo, file, preprocess) in MODELS.items()
    }

    for path in args.inputs:
        samples = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE)
        outdir = args.outdir / path.stem
        outdir.mkdir(parents=True, exist_ok=True)
        for name, (model, preprocess) in models.items():
            probs = predict(model, [preprocess(text) for text in samples["text"]])
            probs.insert(0, "ground_truth", samples["ground_truth"])
            probs.to_csv(outdir / f"{name}.csv", index=False, float_format="%.6g")
            print(f"{path} x {name}: {probs.shape[0]} sentences, {probs.shape[1] - 1} labels -> {outdir / name}.csv")


if __name__ == "__main__":
    main()
