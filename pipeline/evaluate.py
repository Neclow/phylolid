"""Score a zoo language-ID model on the known-language (test) and unseen-language (heldout) sentences of data/splits."""

import re

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ete4 import Tree
from pyglottolog import Glottolog
from torchmetrics import MetricCollection

from src._config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_EXTERN_DIR,
    DEFAULT_REFERENCE_TREE_PROCESSED_DIR,
    DEFAULT_SPLIT_DIR,
    GLOTLIDC_MODELS,
    HELDOUT_FAMILIES,
    NON_LANGUAGE_GLOTTOCODE,
)
from src.data.datasets import TextDataset
from src.lid.metrics import (
    KnownAccuracy,
    KnownHRSeverity,
    KnownIHRSeverity,
    KnownLCAF1Severity,
    KnownMacroF1,
    KnownPathSeverity,
    Severity,
    UnseenHRSeverity,
    UnseenIHRSeverity,
    UnseenLCAF1Severity,
    UnseenOOVRank1,
    UnseenPathSeverity,
    UnseenSameFamily,
    crm,
    lca_f1,
)
from src.models._model_zoo import MODEL_ZOO, load_model

GLOTTOLOG = f"{DEFAULT_EXTERN_DIR}/glottolog"
# Splits -> arms: 2 datasets x 2 arms, each scored apart (notes/plan.md, option b)
ARMS = {"test": "known", "heldout": "unseen"}
# Raw model labels saved per sentence, with their probabilities or scores
TOP_K = 10
# Reference tree of the path severity (an independent reference to Glottolog)
PATH_TREE = "asjp21"


def lineages(rows):
    """Glottocode -> its lineage, from the family down to the languoid itself, from glottolog.csv rows."""
    columns = sorted(
        (column for column in rows.columns if re.fullmatch(r"H\d+", column)),
        key=lambda column: int(column[1:]),
    )
    rows = rows.dropna(subset=["glottocode"]).drop_duplicates("glottocode")
    return {
        row["glottocode"]: tuple(g for g in row[columns] if isinstance(g, str))
        + (row["glottocode"],)
        for _, row in rows.iterrows()
    }


def other_language_tag(file, glottocode, languoids):
    # Files <label>_<source>+<code>: text a source labelled <code>, which GlotLID filed under this label. When <code> is
    # another Glottolog language, the text may be a sister language GlotLID merged in (Dari into Persian)
    for code in file.split("+")[1:]:
        languoid = languoids.get(code)
        if languoid and languoid.level.name == "language" and languoid.id != glottocode:
            return True
    return False


def load_sentences(model_id, languoids):
    """Test and held-out sentences: text, class glottocode, file, dataset and split."""
    sentences = pd.DataFrame(
        [{**item, "split": split} for split in ARMS for item in TextDataset(split)]
    )
    # Checked once per file
    files = sentences[["file", "glottocode"]].drop_duplicates()
    files["tagged"] = [
        other_language_tag(file, glottocode, languoids)
        for file, glottocode in zip(files["file"], files["glottocode"])
    ]
    tagged = sentences.merge(files, on=["file", "glottocode"], how="left")["tagged"]
    print(
        f"Skipped {tagged.sum()} sentences of files tagged with another language: "
        f"{sorted(sentences.loc[tagged, 'file'].unique())}"
    )
    sentences = sentences[~tagged]
    # Models trained on GlotLID-C: scored on Omnilingual only ("--" on GlotLID-C)
    if model_id in GLOTLIDC_MODELS:
        sentences = sentences[sentences["dataset"] != "glotlid-corpus"]
    return sentences.reset_index(drop=True)


def model_classes(model_id, labels):
    """The model's classes (glottocodes, und's last if not one already) and each label's class index."""
    mapping = pd.read_csv(f"{DEFAULT_CACHE_DIR}/glottolog.csv")
    mapping = mapping[mapping["model_id"] == model_id].set_index("label")
    glottocodes = [mapping.loc[label, "glottocode"] for label in labels]
    # Labels of one glottocode (scripts) are one class; und (a row of zeros) is the non-language class
    classes = list(dict.fromkeys(glottocodes + [NON_LANGUAGE_GLOTTOCODE]))
    index = {glottocode: i for i, glottocode in enumerate(classes)}
    return classes, torch.tensor([index[g] for g in glottocodes]), mapping


def class_scores(scores, label_class, n_classes, probabilistic):
    # Labels of one class: probabilities add up; scores (not probabilities) keep the best one
    scores = torch.from_numpy(np.asarray(scores, dtype=np.float32))
    index = label_class.expand(len(scores), -1)
    reduce = "sum" if probabilistic else "amax"
    return torch.zeros(len(scores), n_classes).scatter_reduce(
        1, index, scores, reduce=reduce
    )


def predict(model, sentences, batch_size, label_class, n_classes, und, cost):
    """Top-k raw labels, argmax class and (models with probabilities) CRM class of every sentence."""
    probabilistic = hasattr(model, "predict_proba")
    scores_of = getattr(model, "predict_proba", None) or getattr(
        model, "decision_function"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cost = cost.to(device) if cost is not None else None
    labels = np.array(model.labels())
    texts = sentences["text"].tolist()
    top_labels, top_scores = [None] * len(texts), [None] * len(texts)
    argmax = np.zeros(len(texts), dtype=int)
    crm_pick = np.full(len(texts), -1)
    # Batches of similar lengths: less padding for the GPU models
    order = np.argsort([len(text) for text in texts], kind="stable")
    for start in range(0, len(texts), batch_size):
        batch = order[start : start + batch_size]
        scores = np.asarray(scores_of([texts[i] for i in batch]))
        top = np.argsort(-scores, axis=1)[:, :TOP_K]
        for row, i in enumerate(batch):
            keep = scores[row, top[row]] > 0
            top_labels[i] = labels[top[row][keep]].tolist()
            top_scores[i] = scores[row, top[row][keep]].tolist()
        per_class = class_scores(scores, label_class, n_classes, probabilistic)
        # A row of zeros: the model gave nothing (und)
        empty = per_class.max(dim=1).values <= 0
        argmax[batch] = torch.where(empty, und, per_class.argmax(dim=1)).numpy()
        if probabilistic:
            picks = crm(per_class.to(device), cost).cpu()
            crm_pick[batch] = torch.where(empty, und, picks).numpy()
        if start // batch_size % 100 == 0:
            print(f"{start + len(batch)} / {len(texts)} sentences")
    return top_labels, top_scores, argmax, crm_pick


def metric_collections(true_lineages, n_classes, counted, trees):
    # Severities build their cost tables once; each dataset and decision rule gets a clone. No compute groups: metrics
    # with the same states but different costs must not share them
    known = MetricCollection(
        [
            KnownAccuracy(n_classes),
            KnownMacroF1(n_classes),
            KnownHRSeverity(true_lineages, n_classes),
            KnownIHRSeverity(true_lineages, n_classes, counted),
            KnownLCAF1Severity(true_lineages, n_classes),
            KnownPathSeverity(true_lineages, n_classes, trees),
        ],
        compute_groups=False,
    )
    unseen = MetricCollection(
        [
            UnseenSameFamily(true_lineages, n_classes),
            UnseenOOVRank1(true_lineages, n_classes),
            UnseenHRSeverity(true_lineages, n_classes),
            UnseenIHRSeverity(true_lineages, n_classes, counted),
            UnseenLCAF1Severity(true_lineages, n_classes),
            UnseenPathSeverity(true_lineages, n_classes, trees),
        ],
        compute_groups=False,
    )
    return {"known": known, "unseen": unseen}


def score(collection, preds, target):
    """One row per metric: value, best possible and random references, languages scored, sentences missing."""
    collection = collection.clone()
    collection.update(torch.from_numpy(preds), torch.from_numpy(target))
    rows = []
    for name, metric in collection.items():
        row = {"metric": name, "value": metric.compute().item()}
        if isinstance(metric, Severity):
            references = metric.references()
            row |= {
                "best": references["best"].item(),
                "random": references["random"].item(),
                "languages": int((metric.count > 0).sum()),
                "missing": int(metric.missing),
            }
        else:
            row["languages"] = len(np.unique(target))
        rows.append(row)
    return rows


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ZOO))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Sentences per prediction call (GPU models: mind memory on shared GPUs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(f"{DEFAULT_EVAL_DIR}/zoo/{args.model.replace('/', '--')}")
    if output_dir.exists():
        raise SystemExit(f"{output_dir} exists: move or remove it first")

    print("== Sentences ==")
    languoids = Glottolog(GLOTTOLOG).languoids_by_code()
    sentences = load_sentences(args.model, languoids)
    print(sentences.groupby(["dataset", "split"]).size().to_string())

    print("== Model ==")
    model = load_model(args.model, DEFAULT_CACHE_DIR)
    classes, label_class, mapping = model_classes(args.model, model.labels())
    n_classes = len(classes)
    und = classes.index(NON_LANGUAGE_GLOTTOCODE)
    held = sorted(sentences.loc[sentences["split"] == "heldout", "glottocode"].unique())
    assert not set(held) & set(classes), "a held-out language is one of the classes"

    # Lineages: the model's classes first, then the unseen languages (src/lid/metrics.py)
    dataset_lineages = lineages(pd.read_csv(f"{DEFAULT_DATASET_DIR}/glottolog.csv"))
    model_lineages = lineages(mapping.reset_index())
    model_lineages[NON_LANGUAGE_GLOTTOCODE] = (NON_LANGUAGE_GLOTTOCODE,)
    true_lineages = [model_lineages[g] for g in classes] + [
        dataset_lineages[g] for g in held
    ]
    # ihR counts every glottocode of data/splits, the same set for every model (notes/metrics.md)
    roles = pd.read_csv(f"{DEFAULT_SPLIT_DIR}/languages.csv")
    split_glottocodes = roles.loc[
        (roles["dataset"] == "glotlid-corpus") & (roles["role"] == "train")
        | (roles["role"] == "heldout"),
        "glottocode",
    ].unique()
    counted = [dataset_lineages[g] for g in split_glottocodes]
    trees = {
        family: Tree(
            open(f"{DEFAULT_REFERENCE_TREE_PROCESSED_DIR}/{family}/{PATH_TREE}.nwk"),
            parser=1,
        )
        for family in HELDOUT_FAMILIES
    }
    print(
        f"{len(model.labels())} labels -> {n_classes} classes; {len(held)} unseen languages"
    )

    # Targets: a known class (strict rule: the exact glottocode), an unseen language (>= n_classes), or -1 (left out)
    index = {g: i for i, g in enumerate(classes)} | {
        g: n_classes + i for i, g in enumerate(held)
    }
    sentences["target"] = sentences["glottocode"].map(index).fillna(-1).astype(int)

    print("== Predictions ==")
    # CRM cost between classes: Glottolog, 1 - LCA-hF
    cost = None
    if hasattr(model, "predict_proba"):
        class_lineages = true_lineages[:n_classes]
        cost = 1 - lca_f1(class_lineages, class_lineages)
    top_labels, top_scores, argmax, crm_pick = predict(
        model, sentences, args.batch_size, label_class, n_classes, und, cost
    )

    print("== Scores ==")
    collections = metric_collections(true_lineages, n_classes, counted, trees)
    rules = {"argmax": argmax} | ({"crm": crm_pick} if cost is not None else {})
    rows = []
    for dataset in sorted(sentences["dataset"].unique()):
        for split, arm in ARMS.items():
            mask = (
                (sentences["dataset"] == dataset)
                & (sentences["split"] == split)
                & (sentences["target"] >= 0)
            ).to_numpy()
            if not mask.any():
                continue
            for rule, preds in rules.items():
                for row in score(
                    collections[arm],
                    preds[mask],
                    sentences["target"].to_numpy()[mask],
                ):
                    rows.append({"dataset": dataset, "arm": arm, "rule": rule} | row)
    scores = pd.DataFrame(rows)
    print(scores.to_string(index=False))

    output_dir.mkdir(parents=True)
    scores.to_csv(output_dir / "scores.csv", index=False)
    predictions = sentences.drop(columns="text").assign(
        top_labels=top_labels,
        top_scores=top_scores,
        argmax=[classes[i] for i in argmax],
        crm=[classes[i] if i >= 0 else None for i in crm_pick],
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    print(f"-> {output_dir}")


if __name__ == "__main__":
    main()
