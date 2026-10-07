from typing import Final

DEFAULT_DATA_DIR: Final = "data"
DEFAULT_DATASET_DIR: Final = f"{DEFAULT_DATA_DIR}/datasets"
# Short dataset names -> their folder in DEFAULT_DATASET_DIR
DATASET_ALIASES: Final = {"glotlid": "glotlid-corpus", "omnilingual": "omnilingual-asr"}
# Glottolog families held-out languages come from: Indo-European, Atlantic-Congo, Sino-Tibetan, Afro-Asiatic,
# Austronesian
HELDOUT_FAMILIES: Final = ["indo1319", "atla1278", "sino1245", "afro1255", "aust1307"]
DEFAULT_TREE_DIR: Final = f"{DEFAULT_DATA_DIR}/trees"
DEFAULT_REFERENCE_TREE_DIR: Final = f"{DEFAULT_TREE_DIR}/references"
DEFAULT_REFERENCE_TREE_RAW_DIR: Final = f"{DEFAULT_REFERENCE_TREE_DIR}/raw"
DEFAULT_REFERENCE_TREE_PROCESSED_DIR: Final = f"{DEFAULT_REFERENCE_TREE_DIR}/processed"
DEFAULT_CACHE_DIR: Final = f"{DEFAULT_DATA_DIR}/models"
# Glottocode of non-language labels (undetermined text, no linguistic content), so that scoring works in glottocode
# space only. Glottolog numbers each prefix from 1234, so it can never be a real glottocode
NON_LANGUAGE_GLOTTOCODE: Final = "zxxx0000"

DEFAULT_EXTERN_DIR: Final = "extern"
