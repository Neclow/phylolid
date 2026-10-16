"""Table of our training runs (pipeline/train.py) in data/eval/runs.csv: run id, flags, checkpoint step, evaluated."""

import os
import re

from glob import glob

import pandas as pd
import yaml

from src._config import DEFAULT_EVAL_DIR

# train.py's flags that change the model
FLAGS = [
    "soft_lambda",
    "soft_beta",
    "patristic_weight",
    "quartet_weight",
    "phylonn_weight",
    "lr",
    "batch_size",
    "n_epochs",
    "dim",
    "seed",
]


def runs():
    """One row per run with a checkpoint, in order of start; a flag the run's code did not have yet is left empty."""
    rows = []
    # wandb's folder per run: run-<date>_<time>-<run>
    for run_dir in sorted(glob(f"{DEFAULT_EVAL_DIR}/wandb/run-*")):
        _, started, run = os.path.basename(run_dir).split("-")
        # Runs stopped before their first checkpoint are left out
        checkpoints = glob(f"{DEFAULT_EVAL_DIR}/*/{run}/checkpoints/*.ckpt")
        if not checkpoints:
            continue
        with open(f"{run_dir}/files/config.yaml") as f:
            config = yaml.safe_load(f)
        rows.append(
            {
                "run": run,
                "started": pd.to_datetime(started, format="%Y%m%d_%H%M%S"),
                "step": int(re.findall(r"step=(\d+)", checkpoints[0])[0]),
                "evaluated": os.path.exists(
                    f"{DEFAULT_EVAL_DIR}/trained/{run}/scores.csv"
                ),
                **{flag: config[flag]["value"] for flag in FLAGS if flag in config},
            }
        )
    return pd.DataFrame(rows, columns=["run", "started", "step", "evaluated", *FLAGS])


def main():
    table = runs()
    table.to_csv(f"{DEFAULT_EVAL_DIR}/runs.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
