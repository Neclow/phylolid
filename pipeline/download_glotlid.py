"""Download the GlotLID-C corpus from Hugging Face."""

from huggingface_hub import snapshot_download

from src._config import DEFAULT_DATASET_DIR

# Gated dataset: needs a Hugging Face token with access
REPO = "cis-lmu/glotlid-corpus"
# 2026-04-03, the commit the shelved exploration snapshot came from
REVISION = "63784da4399bb3ea4d04821f7105b31d61ec98bc"
VERSION = "v3.1"
OUTDIR = f"{DEFAULT_DATASET_DIR}/glotlid-corpus"


def main():
    # Files land at OUTDIR/<VERSION>/<label>/<label>_<source>.txt; files already downloaded are skipped
    path = snapshot_download(
        REPO,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=f"{VERSION}/*",
        local_dir=OUTDIR,
    )
    print(f"{REPO}@{REVISION[:7]} {VERSION} -> {path}/{VERSION}")


if __name__ == "__main__":
    main()
