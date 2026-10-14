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
    # Macrolanguages written in one standard, mapped to it as Arabic and Chinese above (label audit, 2026-10-14). No
    # dataset label uses these codes: GlotLID-C has the members (als/aln, nob/nno, lvs/ltg).
    # Albanian (lid.176, CLD3, FUN-LangID sq): standard Albanian is Tosk-based -> Northern Tosk Albanian
    "sqi": "tosk1239",
    # Norwegian (lid.176 and CLD3 no, XLM-V's FLEURS nb_no): Bokmål, Nynorsk having its own nn -> Norwegian Bokmål
    "nor": "norw1259",
    # Latvian (lid.176, CLD3, FUN-LangID lv, XLM-V's FLEURS lv_lv, FLORES lvs_Latn text) -> Standard Latvian
    "lav": "stan1325",
}

# One model's labels whose meaning differs from their code -> glottocode, for that model only (label audit,
# 2026-10-14). Matched on the raw label, before any code conversion (langcodes turns lid.176's sh into Serbian)
MODEL_OVERRIDES = {
    # lid.176 was trained on Wikipedia, whose codes it uses
    "fasttext/lid.176": {
        # The Alemannic Wikipedia (ISO 639-3 als: Tosk Albanian). lid.176 predicts als for 606 of 1,000 Central
        # Alemannic test sentences, and sq for 966 of 1,000 Tosk ones -> Alemannic
        "als": "alem1243",
        # The Serbo-Croatian Wikipedia, apart from sr (Serbian) -> Serbian-Croatian-Bosnian
        "sh": "sout1528",
    },
    # FLORES-200's kon_Latn text is Kituba (FLORES+ relabelled it ktu_Latn); its tzm_Tfng text was relabelled zgh_Tfng
    "facebook/fasttext-language-identification": {
        "kon_Latn": "kitu1246",
        "tzm_Tfng": "stan1324",
    },
    # As NLLB: OpenLID-v2 calls the same classes ktu_Latn and zgh_Tfng
    "laurievb/OpenLID": {"kon_Latn": "kitu1246", "tzm_Tfng": "stan1324"},
    # FLEURS Fula (ff_sn) transcripts are FLORES fuv_Latn text -> Nigerian Fulfulde (Glottolog: Hausa States Fulfulde)
    "juliensimon/xlm-v-base-language-id": {"Fula": "nige1253"},
    # franc's hna profile is Gen, a Gbe language (franc-all now lists it as gej), not Mina of Cameroon (Chadic)
    "cyb3rk0tik/pyfranc": {"hna": "genn1243"},
    # Its nrm_Latn class is Norman (Wikipedia's nrm, predicted at p ≈ 1 on Norman text), not Narom (Austronesian)
    "cis-lmu/glotlid": {"nrm_Latn": "norm1245"},
    # Matched by FUN-LangID's own n-gram profiles against GlotLID-C text
    "google-research/fun_langid": {
        # Central Kanuri text (FLORES and NLLB-Seed), not Manga Kanuri
        "kr": "cent2050",
        "kr-Arab": "cent2050",
        # Dogri text (SMOL), apart from its xnr class, which is Kangri text (UD)
        "doi": "dogr1250",
        "doi-Arab": "dogr1250",
        "xnr": "kang1280",
        # Marwari, apart from its raj (Rajasthani) class
        "mwr": "marw1260",
        # The region picks out Cypriot Greek, a language of its own in Glottolog
        "el-CY": "cypr1249",
        # Hmong (RPA spelling) is Hmong Daw, as its mww-CN: kept together (plain hmn has no languoid)
        "hmn": "hmon1333",
    },
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


def map_label(label, languoids, relabels, model_overrides=None):
    """Script, how the glottocode was found, and the Glottolog columns of a label (empty if it has no languoid)."""
    language, script = split_label(label)
    # Not a language: the model abstains (undetermined text, no linguistic content)
    if language in NON_LANGUAGE_CODES:
        return {
            "script": script,
            "via": "non-language",
            "glottocode": NON_LANGUAGE_GLOTTOCODE,
        }
    if model_overrides and label in model_overrides:
        languoid, via = languoids.get(model_overrides[label]), "model override"
    else:
        iso, via = iso639_3(language)
        # Codes that GlotLID files under another label follow GlotLID, for consistency (e.g. est -> ekk, tgl -> fil)
        if iso in relabels:
            iso, via = relabels[iso], "glotlid"
        elif iso in OVERRIDES:
            via = "override"
        # A languoid can be a dialect or a family rather than a language. It is kept at that level: moving dialects up
        # to their language would merge labels (e.g. nrf_Latn into fra_Latn)
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


def map_model_labels(labels, languoids, relabels, model_overrides):
    """A model's labels -> map_label rows; a GlotLID refiling that would merge two of the model's codes is undone."""
    mapped = {
        label: map_label(label, languoids, relabels, model_overrides)
        for label in labels
    }
    # Codes per glottocode: two codes on one glottocode merge two classes the model keeps apart
    codes = {}
    for label, row in mapped.items():
        codes.setdefault(row.get("glottocode"), set()).add(split_label(label)[0])
    for label, row in mapped.items():
        # GlotLID refiled the data, but this model keeps both codes: the label keeps its own code's meaning (e.g. NLLB's
        # prs_Arab is Dari, not Western Farsi as GlotLID files it)
        if row["via"] == "glotlid" and len(codes[row.get("glottocode")]) > 1:
            mapped[label] = map_label(label, languoids, {}, model_overrides)
    return mapped


def print_merges(df):
    # Different codes of one model still on one glottocode, i.e. one class (script variants are not listed)
    labels = df[df["glottocode"].notna() & (df["via"] != "non-language")]
    for (model_id, glottocode), group in labels.groupby(["model_id", "glottocode"]):
        if group["label"].map(lambda label: split_label(label)[0]).nunique() > 1:
            print(f"{model_id}: {' '.join(group['label'])} -> {glottocode} (one class)")


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
        mapped = map_model_labels(
            labels, languoids, relabels, MODEL_OVERRIDES.get(model_id, {})
        )
        rows += [
            {"model_id": model_id, "label": label, **mapped[label]} for label in labels
        ]
    write(rows, "model_id", f"{DEFAULT_CACHE_DIR}/glottolog.csv")
    print_merges(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
