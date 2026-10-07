"""Map every label of the datasets and of the model zoo to its Glottolog languoid, script and lineage."""

import re

from glob import glob
from pathlib import Path

import langcodes
import pandas as pd

from pyglottolog import Glottolog

from src._config import (
    DATASET_ALIASES,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_EXTERN_DIR,
    NON_LANGUAGE_GLOTTOCODE,
)
from src.models._model_zoo import MODEL_ZOO, load_model

# Local Glottolog clone (git submodule)
GLOTTOLOG = f"{DEFAULT_EXTERN_DIR}/glottolog"

# Codes that cover several languages, or that Glottolog lacks, and that GlotLID does not relabel (glotlid_relabels)
# -> glottocode, or ISO 639-3 code of the language that represents them. Where one language represents them, it is
# that language; otherwise the lowest Glottolog group that covers them. They apply to every dataset and model. Sources:
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
    # Macrolanguage Arabic (e.g. lid.176 and CLD3's ar, XLM-V's "Arabic"): written web and Wikipedia text is mostly
    # Modern Standard Arabic, and FLEURS Arabic is MSA read by Egyptian speakers (as in phylaudio) -> Standard Arabic
    "ara": "stan1318",
    # Macrolanguage Malay (e.g. ms, "Malay"): these models label Indonesian separately (id), and phylaudio maps FLEURS
    # Malay the same way -> Standard Malay
    "msa": "stan1306",
    # Macrolanguage Chinese (zh, zho_Hans, zho_Hant): GlotLID dropped it for its individual languages (languages-v3.md),
    # maps FLORES Chinese to cmn_Hani (flores.json), and Mandarin is most of its Chinese text -> Mandarin
    "zho": "cmn",
    # Macrolanguage Konkani: GlotLID files Konkani (knn) data under gom (+knn), its only Konkani -> Goan Konkani
    "kok": "gom",
    # Macrolanguage Syriac: GlotLID files Classical Syriac (syc) and Aramaic (arc) data under aii (+syc, +arc), its only
    # Syriac -> Assyrian Neo-Aramaic
    "syr": "aii",
    # Macrolanguage Inuktitut: Eastern Canadian Inuktitut, in Inuktitut's syllabics, is most of GlotLID's Inuktitut
    # (ike_Cans; ikt_Latn is small) -> Eastern Canadian Inuktitut
    "iku": "ike",
    # ISO 639-2 collective code "Bihari languages" (lid.176's bh): lid.176 was trained on Wikipedia, and bh.wikipedia is
    # the Bhojpuri Wikipedia -> Bhojpuri
    "bih": "bho",
    # Macrolanguage Quechua: 27 Quechua labels in GlotLID-C, none dominant -> Quechuan, which covers all its ISO members
    # but Chilean Quechua (under Glottolog's Bookkeeping)
    "que": "quec1387",
    # Macrolanguage Luyia -> Greater Luyia, which covers all its ISO members
    "luy": "grea1291",
    # Macrolanguage Rajasthani -> Glottolog's Rajasthani group (its ISO members' lowest common group, Midlands
    # Indo-Aryan, is far broader)
    "raj": "raja1256",
    # Macrolanguage Marwari -> Rajasthani, the lowest group that covers all its ISO members
    "mwr": "raja1256",
    # ISO 639-5 collective code "Berber languages" -> Berber
    "ber": "berb1260",
}

# Codes of non-language labels: undetermined text (und), no linguistic content (zxx)
NON_LANGUAGE_CODES = {"und", "zxx"}


def split_label(label):
    """Language and script of a label: kal_Latn -> (kal, Latn), bg-Latn -> (bg, Latn), en -> (en, None).

    Labels that are not codes (language names, e.g. Afrikaans) are kept whole, without a script.
    """
    if not re.match(r"[a-z]{2,3}([_-]|$)", label):
        return label, None
    language, *subtags = re.split(r"[_-]", label)
    # Scripts are four-letter ISO 15924 codes (Latn, Cyrl); other subtags (regions, private use) are dropped
    script = next((tag for tag in subtags if re.fullmatch(r"[A-Z][a-z]{3}", tag)), None)
    return language, script


def iso639_3(language):
    """ISO 639-3 code of a three-letter code, a two-letter code or a language name, and how it was found."""
    if re.fullmatch(r"[a-z]{3}", language):
        return language, "iso639-3"
    try:
        if re.fullmatch(r"[a-z]{2}", language):
            # Also replaces deprecated codes, e.g. iw -> he
            return langcodes.Language.get(language).to_alpha3(), "iso639-1"
        return langcodes.find(language).to_alpha3(), "name"
    except LookupError:
        return None, None


def glotlid_relabels():
    """GlotLID's own relabellings: code -> ISO 639-3 code of the GlotLID-C label it filed that code's data under.

    A file <label>/<label>_<source>+<code>.txt holds data that a source labelled <code> (e.g. ekk_Latn/..+est). Codes
    that are GlotLID-C labels themselves keep their own mapping, since GlotLID swaps some both ways (kng, ktu).
    """
    corpus = f"{DEFAULT_DATASET_DIR}/{DATASET_ALIASES['glotlid']}"
    labels = {Path(path).name.split("_")[0] for path in glob(f"{corpus}/*/*/")}
    relabels = {}
    for path in glob(f"{corpus}/*/*/*+*.txt"):
        label = Path(path).parent.name.split("_")[0]
        # Tags that are not ISO 639-3 codes (e.g. apcnort) are skipped
        for code in Path(path).stem.split("+")[1:]:
            if re.fullmatch(r"[a-z]{3}", code) and code not in labels:
                relabels[code] = label
    return relabels


def map_label(label, languoids, relabels):
    """Script, how the glottocode was found, and the Glottolog columns of a label (empty if it has no languoid)."""
    language, script = split_label(label)
    # Not a language: the model abstains (undetermined text, no linguistic content)
    if language in NON_LANGUAGE_CODES:
        return {
            "script": script,
            "via": "non-language",
            "glottocode": NON_LANGUAGE_GLOTTOCODE,
        }
    iso, via = iso639_3(language)
    # Codes that GlotLID files under another label follow GlotLID, for consistency (e.g. est -> ekk, tgl -> fil)
    if iso in relabels:
        iso, via = relabels[iso], "glotlid"
    elif iso in OVERRIDES:
        via = "override"
    # A languoid can be a dialect or a family rather than a language. It is kept at that level: moving dialects up to
    # their language would merge labels (e.g. nrf_Latn into fra_Latn)
    languoid = languoids.get(OVERRIDES.get(iso, iso)) if iso else None
    if languoid is None:
        return {"script": script, "via": via}
    return {
        "script": script,
        "via": via,
        "name": languoid.name,
        "glottocode": languoid.id,
        "level": languoid.level.name if languoid.level else None,
        # Ancestors' glottocodes, from H0 (top-level family) down to the closest one
        **{
            f"H{i}": glottocode for i, (_, glottocode, _) in enumerate(languoid.lineage)
        },
    }


def write(rows, source_column, output_file):
    df = pd.DataFrame(rows)
    lineage_columns = sorted(
        (column for column in df.columns if re.fullmatch(r"H\d+", column)),
        key=lambda column: int(column[1:]),
    )
    columns = [source_column, "label", "script", "via", "name", "glottocode", "level"]
    df = df.reindex(columns=columns + lineage_columns)
    df.to_csv(output_file, index=False)
    for source, group in df.groupby(source_column, sort=False):
        non_language = group["via"] == "non-language"
        missing = group.loc[group["glottocode"].isna() & ~non_language, "label"]
        print(
            f"{source}: {len(group)} labels ({non_language.sum()} non-language), "
            f"{len(missing)} without a glottocode: {missing.tolist()}"
        )
    print(f"{len(df)} labels -> {output_file}")


def main():
    # ISO 639-3 codes and glottocodes -> languoid
    languoids = Glottolog(GLOTTOLOG).languoids_by_code()
    relabels = glotlid_relabels()

    print("== Dataset labels ==")
    # Dataset labels are the folders <dataset>/<version>/<label>/, named <ISO 639-3>_<script> (e.g. kal_Latn).
    # glob skips hidden folders (.cache)
    rows = [
        {
            "dataset": Path(dataset_dir).name,
            "label": label,
            **map_label(label, languoids, relabels),
        }
        for dataset_dir in sorted(glob(f"{DEFAULT_DATASET_DIR}/*/"))
        for label in sorted(Path(path).name for path in glob(f"{dataset_dir}*/*/"))
    ]
    write(rows, "dataset", f"{DEFAULT_DATASET_DIR}/glottolog.csv")

    print("== Model labels ==")
    rows = []
    for model_id in MODEL_ZOO:
        print(f"Loading model: {model_id}")
        labels = load_model(model_id, DEFAULT_CACHE_DIR).labels()
        rows += [
            {
                "model_id": model_id,
                "label": label,
                **map_label(label, languoids, relabels),
            }
            for label in labels
        ]
    write(rows, "model_id", f"{DEFAULT_CACHE_DIR}/glottolog.csv")


if __name__ == "__main__":
    main()
