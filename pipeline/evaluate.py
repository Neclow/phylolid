"""Score a zoo language-ID model on the known-language (test) and unseen-language (heldout) sentences of data/splits."""

import logging
import os
import re

from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from ete4 import Tree
from torchmetrics import MetricCollection

from src._config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_N_THREADS,
    DEFAULT_REFERENCE_TREE_PROCESSED_DIR,
    DEFAULT_SPLIT_DIR,
    GLOTLIDC_MODELS,
    HELDOUT_FAMILIES,
    NON_LANGUAGE_GLOTTOCODE,
)
from src.data.datasets import TextDataset
from src.lid.features import Features
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
from src.lid.model import Predictor
from src.models._model_zoo import MODEL_ZOO, load_model

# GlotLID-C files (<label>_<source>+<tag>.txt) of a sister language's text, which GlotLID merged under the label:
# skipped at test, as a model that recognises them would be scored wrong. Not GlotLID's corrections of mislabelled
# sources (e.g. ktu_Latn +kng), which hold the label's own text. Evidence in notes/label_audit.md
MERGED_TAGS = {"tet_Latn": ["tdt"], "aii_Syrc": ["syc", "arc"], "fas_Arab": ["prs"]}
# ISO 15924 variants written in their script's Unicode characters: a label in a variant covers text in the script.
# Kana covers Japanese text, tagged Jpan (Han + Hiragana + Katakana)
SCRIPT_VARIANTS = {
    "Hans": "Hani",
    "Hant": "Hani",
    "Cyrs": "Cyrl",
    "Hira": "Jpan",
    "Kana": "Jpan",
}
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


def merged(file):
    label = "_".join(file.split("_")[:2])
    return any(tag in MERGED_TAGS.get(label, []) for tag in file.split("+")[1:])


def load_sentences(model_id):
    """Test and held-out sentences: text, class glottocode, file, dataset and split."""
    sentences = pd.DataFrame(
        [{**item, "split": split} for split in ARMS for item in TextDataset(split)]
    )
    skipped = sentences["file"].isin(
        [file for file in sentences["file"].unique() if merged(file)]
    )
    logging.info(
        f"Skipped {skipped.sum()} sentences of merged files: "
        f"{sorted(sentences.loc[skipped, 'file'].unique())}"
    )
    sentences = sentences[~skipped]
    # Models trained on GlotLID-C: scored on Omnilingual only ("--" on GlotLID-C)
    if model_id in GLOTLIDC_MODELS:
        sentences = sentences[sentences["dataset"] != "glotlid-corpus"]
    return sentences.reset_index(drop=True)


def zoo_mapping(model_id):
    """A zoo model's labels -> glottocode, scripts and lineage (pipeline/map_glottolog.py)."""
    mapping = pd.read_csv(f"{DEFAULT_CACHE_DIR}/glottolog.csv")
    return mapping[mapping["model_id"] == model_id].set_index("label")


def trained_mapping(classes):
    """Our model's labels, glottocodes, in zoo_mapping's form: lineages from the datasets' mapping, scripts those of
    each class's training files (<label>_<source>.txt, labels <iso>_<Script>)."""
    rows = pd.read_csv(f"{DEFAULT_DATASET_DIR}/glottolog.csv")
    rows = rows.drop_duplicates("glottocode").set_index("glottocode")
    lineage = [column for column in rows.columns if re.fullmatch(r"H\d+", column)]
    scripts = [
        {
            os.path.basename(path).split("_")[1]
            for path in glob(f"{DEFAULT_SPLIT_DIR}/train/{g}/*.txt")
        }
        for g in classes
    ]
    mapping = rows.loc[classes, lineage].assign(
        glottocode=classes, script=[" ".join(sorted(s)) for s in scripts]
    )
    return mapping.rename_axis("label")


def model_classes(mapping, labels):
    """The model's classes (glottocodes, und's last if not one already) and each label's class index."""
    glottocodes = [mapping.loc[label, "glottocode"] for label in labels]
    # Labels of one glottocode (scripts) are one class; und (a row of zeros) is the non-language class
    classes = list(dict.fromkeys(glottocodes + [NON_LANGUAGE_GLOTTOCODE]))
    index = {glottocode: i for i, glottocode in enumerate(classes)}
    return classes, torch.tensor([index[g] for g in glottocodes])


def label_scripts(mapping):
    """Glottocode -> the scripts its labels cover, or None when one's are unknown (it covers every script)."""
    # Space-separated scripts per label (pipeline/map_glottolog.py)
    scripts = mapping["script"].str.split()
    return {
        glottocode: (
            None
            if group.isna().any()
            else {SCRIPT_VARIANTS.get(s, s) for label in group for s in label}
        )
        for glottocode, group in scripts.groupby(mapping["glottocode"])
    }


def class_scores(scores, label_class, n_classes, probabilistic):
    # Labels of one class: probabilities add up; scores (not probabilities) keep the best one
    scores = torch.from_numpy(np.asarray(scores, dtype=np.float32))
    index = label_class.expand(len(scores), -1)
    reduce = "sum" if probabilistic else "amax"
    return torch.zeros(len(scores), n_classes).scatter_reduce(
        1, index, scores, reduce=reduce
    )


def scores_of(model):
    """The model's scores over its labels: its probabilities if it has them, else its decision scores."""
    return getattr(model, "predict_proba", None) or getattr(model, "decision_function")


# A worker process's own copy of the model, when batches are scored in parallel
worker_model = None


def load_worker(model_id):
    global worker_model
    worker_model = load_model(model_id, DEFAULT_CACHE_DIR)


def worker_scores(texts):
    return scores_of(worker_model)(texts)


def predict(
    model, sentences, batch_size, label_class, n_classes, und, cost, device, n_threads
):
    """Top-k raw labels, argmax class and (models with probabilities) CRM class of every sentence."""
    probabilistic = hasattr(model, "predict_proba")
    cost = cost.to(device) if cost is not None else None
    labels = np.array(model.labels())
    texts = sentences["text"].tolist()
    top_labels, top_scores = [None] * len(texts), [None] * len(texts)
    argmax = np.zeros(len(texts), dtype=int)
    crm_pick = np.full(len(texts), -1)
    # Batches of similar lengths: less padding for the GPU models
    order = np.argsort([len(text) for text in texts], kind="stable")
    batches = [
        order[start : start + batch_size] for start in range(0, len(texts), batch_size)
    ]
    batch_texts = ([texts[i] for i in batch] for batch in batches)
    # Batches scored by n_threads processes, each with its own copy of the model, results in order; PyTorch models
    # (XLM-V, ConLID) score them in this one, on device
    pool = (
        Pool(n_threads, initializer=load_worker, initargs=(model.model_id,))
        if n_threads > 1 and not isinstance(model.model, torch.nn.Module)
        else None
    )
    all_scores = (
        pool.imap(worker_scores, batch_texts)
        if pool
        else map(scores_of(model), batch_texts)
    )
    for n, (batch, scores) in enumerate(zip(batches, all_scores)):
        scores = np.asarray(scores)
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
        if n % 100 == 0:
            logging.info(f"{n * batch_size + len(batch)} / {len(texts)} sentences")
    if pool:
        pool.close()
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
    # A zoo model, or one we trained
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", choices=sorted(MODEL_ZOO), help="Zoo model")
    model.add_argument(
        "--checkpoint",
        help="Lightning checkpoint of a model we trained (pipeline/train.py)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Sentences per prediction call (GPU models: mind memory on shared GPUs)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device of the PyTorch models (XLM-V, ConLID) and of CRM, e.g. cuda:1 or cpu",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=DEFAULT_N_THREADS,
        help="CPU processes scoring batches, each with its own copy of the model (PyTorch models use one)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Outputs in zoo/<org>--<name>, or trained/<run>, the run being the checkpoint's folder
    # (<project>/<run>/checkpoints/<file>.ckpt)
    model_id = args.model or args.checkpoint
    if args.model:
        name, kind = args.model.replace("/", "--"), "zoo"
    else:
        name = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))
        kind = "trained"
    output_dir = f"{DEFAULT_EVAL_DIR}/{kind}/{name}"
    if os.path.exists(output_dir):
        raise SystemExit(f"{output_dir} exists: move or remove it first")
    # To the terminal and to a log file per model
    os.makedirs(f"{DEFAULT_EVAL_DIR}/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{DEFAULT_EVAL_DIR}/logs/{name}.log"),
        ],
    )

    sentences = load_sentences(model_id)
    logging.info(
        "Sentences per dataset and split:\n"
        + sentences.groupby(["dataset", "split"]).size().to_string()
    )

    logging.info(f"Loading {model_id} ({args.device})")
    if args.model:
        model = load_model(args.model, DEFAULT_CACHE_DIR, args.device)
        mapping = zoo_mapping(args.model)
    else:
        features = Features.from_json(f"{DEFAULT_SPLIT_DIR}/vocab.json")
        model = Predictor(args.checkpoint, features, args.device)
        mapping = trained_mapping(model.labels())
    classes, label_class = model_classes(mapping, model.labels())
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
    logging.info(
        f"{len(model.labels())} labels -> {n_classes} classes; {len(held)} unseen languages"
    )

    # Targets: a known class (strict rule: the exact glottocode), an unseen language (>= n_classes), or -1 (left out)
    index = {g: i for i, g in enumerate(classes)} | {
        g: n_classes + i for i, g in enumerate(held)
    }
    sentences["target"] = sentences["glottocode"].map(index).fillna(-1).astype(int)
    # Strict rule, by script: a known class's sentences in a script none of its labels covers are left out too. Files
    # are <iso>_<Script>_<source>
    scripts = label_scripts(mapping)
    sentence_scripts = sentences["file"].str.split("_").str[1].replace(SCRIPT_VARIANTS)
    other_script = np.array(
        [
            0 <= target < n_classes
            and scripts[glottocode] is not None
            and script not in scripts[glottocode]
            for glottocode, script, target in zip(
                sentences["glottocode"], sentence_scripts, sentences["target"]
            )
        ]
    )
    sentences.loc[other_script, "target"] = -1
    logging.info(
        f"Left out {other_script.sum()} known-class sentences in a script the model's labels do not cover: "
        f"{sorted(set(sentences.loc[other_script, 'glottocode'] + '_' + sentence_scripts[other_script]))}"
    )

    # CRM cost between classes: Glottolog, 1 - LCA-hF
    cost = None
    if hasattr(model, "predict_proba"):
        class_lineages = true_lineages[:n_classes]
        cost = 1 - lca_f1(class_lineages, class_lineages)
    top_labels, top_scores, argmax, crm_pick = predict(
        model,
        sentences,
        args.batch_size,
        label_class,
        n_classes,
        und,
        cost,
        args.device,
        args.n_threads,
    )

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
    logging.info("Scores:\n" + scores.to_string(index=False))

    os.makedirs(output_dir)
    scores.to_csv(f"{output_dir}/scores.csv", index=False)
    predictions = sentences.drop(columns="text").assign(
        top_labels=top_labels,
        top_scores=top_scores,
        argmax=[classes[i] for i in argmax],
        crm=[classes[i] if i >= 0 else None for i in crm_pick],
    )
    predictions.to_parquet(f"{output_dir}/predictions.parquet", index=False)
    logging.info(f"-> {output_dir}")


if __name__ == "__main__":
    main()
