"""Download GlotLID-C v3.1 (cis-lmu/glotlid-corpus, gated on Hugging Face) into data/datasets/glotlid-corpus/.

Files land at data/datasets/glotlid-corpus/v3.1/<label>/<label>_<source>.txt (1,939 labels, ~44 GB).
Files already downloaded are skipped.

Example:
    pixi run download_glotlid
"""
from huggingface_hub import snapshot_download

from src._config import DEFAULT_DATASET_DIR

REPO = "cis-lmu/glotlid-corpus"
REVISION = "63784da4399bb3ea4d04821f7105b31d61ec98bc"  # 2026-04-03, the commit data/datasets/_glotlid-corpus came from
VERSION = "v3.1"
OUTDIR = f"{DEFAULT_DATASET_DIR}/glotlid-corpus"


def main():
    path = snapshot_download(
        REPO, repo_type="dataset", revision=REVISION, allow_patterns=f"{VERSION}/*", local_dir=OUTDIR
    )
    print(f"{REPO}@{REVISION[:7]} {VERSION} -> {path}/{VERSION}")


if __name__ == "__main__":
    main()
