from typing import Final

DEFAULT_DATA_DIR: Final = "data"
DEFAULT_DATASET_DIR: Final = f"{DEFAULT_DATA_DIR}/datasets"
# Short dataset names -> their folder in DEFAULT_DATASET_DIR
DATASET_ALIASES: Final = {"glotlid": "glotlid-corpus", "omnilingual": "omnilingual-asr"}
# Glottolog families held-out languages come from: Indo-European, Atlantic-Congo, Sino-Tibetan, Afro-Asiatic,
# Austronesian, Otomanguean
HELDOUT_FAMILIES: Final = [
    "indo1319",
    "atla1278",
    "sino1245",
    "afro1255",
    "aust1307",
    "otom1299",
]
# Glottolog's pseudo-families: top-level groups that are not genealogical (bookkeeping, unclassifiable, pidgin, mixed,
# artificial and sign languages, speech registers, unattested). Their members are not relatives of each other
PSEUDO_FAMILIES: Final = [
    "book1242",
    "uncl1493",
    "pidg1258",
    "mixe1287",
    "arti1236",
    "sign1238",
    "spee1234",
    "unat1236",
]
DEFAULT_TREE_DIR: Final = f"{DEFAULT_DATA_DIR}/trees"
DEFAULT_REFERENCE_TREE_DIR: Final = f"{DEFAULT_TREE_DIR}/references"
DEFAULT_REFERENCE_TREE_RAW_DIR: Final = f"{DEFAULT_REFERENCE_TREE_DIR}/raw"
DEFAULT_REFERENCE_TREE_PROCESSED_DIR: Final = f"{DEFAULT_REFERENCE_TREE_DIR}/processed"
DEFAULT_CACHE_DIR: Final = f"{DEFAULT_DATA_DIR}/models"
# Zoo models trained on GlotLID-C: a GlotLID-C language they know can still be held out, and they are scored on
# Omnilingual only ("--" on GlotLID-C, whose test sentences they may have trained on)
GLOTLIDC_MODELS: Final = ["cis-lmu/glotlid", "epfl-nlp/ConLID"]
# Train, dev, test and held-out sentences, one folder each, and the role of every dataset label (languages.csv)
DEFAULT_SPLIT_DIR: Final = f"{DEFAULT_DATA_DIR}/splits"
# Scores and per-sentence predictions of each model: zoo/<org>--<name>/, trained/<run>/
DEFAULT_EVAL_DIR: Final = f"{DEFAULT_DATA_DIR}/eval"
# Our training runs, one row each: wandb run id, train.py's flags, whether evaluate.py has scored the run
DEFAULT_RUNS_FILE: Final = f"{DEFAULT_EVAL_DIR}/runs.csv"
# Glottocode of non-language labels (undetermined text, no linguistic content), so that scoring works in glottocode
# space only. Glottolog numbers each prefix from 1234, so it can never be a real glottocode
NON_LANGUAGE_GLOTTOCODE: Final = "zxxx0000"

DEFAULT_EXTERN_DIR: Final = "extern"
# CPU processes for parallel work (the machine is shared)
DEFAULT_N_THREADS: Final = 16
