"""Glottolog lineages of the labels mapped by pipeline/map_glottolog.py."""

import re


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
