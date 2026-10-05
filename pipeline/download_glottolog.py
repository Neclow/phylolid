"""Map each label of a downloaded dataset to its Glottolog languoid and lineage, as in phylaudio's download_glottolog.py.

Labels are the folders data/datasets/<dataset>/<version>/<label>/, named <ISO 639-3>_<script>
(e.g. glotlid-corpus/v3.1/kal_Latn/), so run the dataset's download script first. The ISO code is
looked up in a local Glottolog clone (extern/glottolog, tag v5.3).

Writes data/datasets/<dataset>/glottolog.csv, one row per label: name, glottocode, level, and the
glottocodes of its ancestors, H0 (top-level family) down to the closest one. Some ISO codes are a
Glottolog dialect or family rather than a language; they are kept at that level, since moving
dialects up to their language would merge labels (e.g. nrf_Latn into fra_Latn). Some ISO codes are
mapped by hand (OVERRIDES). Labels whose ISO code is still not in Glottolog are written with an empty
glottocode and printed.

Example:
    pixi run download_glottolog
"""
import argparse
from glob import glob
from pathlib import Path

import pandas as pd
from pyglottolog import Glottolog

from src._config import DEFAULT_DATASET_DIR

GLOTTOLOG = "extern/glottolog"

# GlotLID-C labels that cover several languages, or whose ISO code Glottolog lacks -> glottocode. Except fas,
# this is the lowest Glottolog group that covers them. Sources:
#   languages-v3.md: https://github.com/cisnlp/GlotLID/blob/main/languages-v3.md
#   flores.json:     https://github.com/cisnlp/GlotLID/blob/main/assets/map/flores.json
#   +tags:           files v3.1/<label>/<label>_<source>+<code>.txt hold data a source labelled <code>
#   README:          https://huggingface.co/datasets/cis-lmu/glotlid-corpus
OVERRIDES = {
    # Iranian Persian (pes) + Dari (prs), merged "due to the close relationship of Dari and Iranian Persian"
    # (languages-v3.md, flores.json, +pes/+prs) -> Western Farsi, as for FLEURS Persian in phylaudio
    "fas": "west2369",
    # Campidanese (sro) + Logudorese (src); Sassarese (sdc) has its own label (languages-v3.md) -> their group, sard1257
    "srd": "sard1257",
    # Macrolanguage kept without its individual languages, Guinea (gkp) and Liberia (xpe) Kpelle (languages-v3.md)
    # -> Kpelle
    "kpe": "kpel1252",
    # Macrolanguage Baluchi; Southern Balochi (bcc) data relabelled bal "for better classification" (README, +bcc)
    # -> Balochic
    "bal": "balo1260",
    # ISO 639-5 collective code "Nahuatl languages"; 16 Nahuatl languages have their own labels -> Aztec
    "nah": "azte1234",
    # ISO 639-5 collective code "Otomian languages"; 5 Otomi languages have their own labels -> Otomian
    "oto": "otom1297",
    # Middle Newar, a historical language, separate from Newari (new_Deva) since GlotLID v1 (paper, appendix B).
    # Not in Glottolog -> Newar, the group above Kathmandu Valley Newari (newa1246, which new_Deva maps to)
    "nwx": "newa1247",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help=f"folder in {DEFAULT_DATASET_DIR}/, e.g. glotlid-corpus")
    args = parser.parse_args()

    dataset_dir = f"{DEFAULT_DATASET_DIR}/{args.dataset}"
    labels = sorted(Path(path).name for path in glob(f"{dataset_dir}/*/*/"))  # glob skips hidden folders (.cache)
    if not labels:
        raise FileNotFoundError(f"No label folders in {dataset_dir}/<version>/. Download the dataset first.")
    print(f"Found {len(labels)} labels")

    languoids = Glottolog(GLOTTOLOG).languoids_by_code()  # ISO 639-3 codes and glottocodes -> languoid

    rows = []
    for label in labels:
        iso = label.split("_")[0]
        languoid = languoids.get(OVERRIDES.get(iso, iso))
        if languoid is None:
            rows.append({"label": label})
            continue
        row = {"label": label, "name": languoid.name, "glottocode": languoid.id, "level": languoid.level.name}
        row |= {f"H{i}": glottocode for i, (_, glottocode, _) in enumerate(languoid.lineage)}
        rows.append(row)

    df = pd.DataFrame(rows)
    output_file = f"{dataset_dir}/glottolog.csv"
    df.to_csv(output_file, index=False)
    print(df["level"].value_counts(dropna=False).to_string())
    print("not in Glottolog:", df.loc[df["glottocode"].isna(), "label"].tolist())
    print(f"{len(df)} labels -> {output_file}")


if __name__ == "__main__":
    main()
