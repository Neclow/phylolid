# Pre-trained models

Off-the-shelf language-ID models of the model zoo (`src/models/_model_zoo.py`),
fetched by `pixi run download_models`. Their languages either define the known
set or serve for evaluation only.

| Model                                       | Role            | Labels | Languages | Where its files are                                                              |
| ------------------------------------------- | --------------- | -----: | --------: | -------------------------------------------------------------------------------- |
| `fasttext/lid.176`                          | known set       |    176 |       176 | `fasttext--lid.176/lid.176.bin`                                                  |
| `facebook/fasttext-language-identification` | known set       |    218 |       211 | `models--facebook--fasttext-language-identification/`                            |
| `laurievb/OpenLID`                          | known set       |    201 |       195 | `models--laurievb--OpenLID/`                                                     |
| `laurievb/OpenLID-v2`                       | known set       |    200 |       194 | `models--laurievb--OpenLID-v2/`                                                  |
| `HPLT/OpenLID-v3`                           | known set       |    195 |       187 | `models--HPLT--OpenLID-v3/`                                                      |
| `google/cld3`                               | known set       |    109 |       103 | not here: compiled into the `gcld3` package of the pixi environment              |
| `juliensimon/xlm-v-base-language-id`        | known set       |    102 |       102 | `models--juliensimon--xlm-v-base-language-id/`                                   |
| `cyb3rk0tik/pyfranc`                        | known set       |    380 |       380 | not here: `data.py` of the `pyfranc` package of the pixi environment             |
| `cis-lmu/glotlid`                           | evaluation only |  2,102 |     1,874 | `models--cis-lmu--glotlid/`                                                      |
| `epfl-nlp/ConLID`                           | evaluation only |  2,099 |     1,871 | `models--epfl-nlp--ConLID/` (weights; its code is the `extern/ConLID` submodule) |
| `google-research/fun_langid`                | evaluation only |  1,647 |     1,552 | `google-research--fun_langid/fun_langid.py`                                      |

- Labels: the model's output classes, as listed in its own files (CLD3:
  `src/task_context_params.cc` of `google/cld3`). A label can be a language with
  a script or region (e.g. `srp_Cyrl`, `bg-Latn`).
- Languages: distinct language codes among the labels, without the non-language
  labels (`und_*`, `zxx_*`). Codes are as each model writes them (ISO 639-1 or
  639-3, BCP-47, or language names for XLM-V), not yet mapped to ISO 639-3.
- `models--<org>--<name>/` folders are Hugging Face caches at the commit pinned
  in the registry; their files live in the shared `blobs/` folder.
- Models downloaded from a URL get a folder named the same way,
  `<org>--<name>/`.
- CLD3 and pyfranc are installed with the pixi environment: `pixi.lock` pins
  their versions.
- `glottolog.csv`, written by `pixi run map_glottolog`: every model label
  (column `model_id`) with its script, Glottolog languoid and lineage, and how
  its glottocode was found (`via`). Same columns as
  `data/datasets/glottolog.csv`.
