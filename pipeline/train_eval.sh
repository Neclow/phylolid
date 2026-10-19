#!/usr/bin/env bash
# Train one run, then evaluate it.
# The run's wandb id is set here (WANDB_RUN_ID), so that the evaluation scores
# this run even when another GPU finishes one at the same time.
# A run stopped with Ctrl-C is not evaluated.
# Usage: pixi run train_eval <device> [train.py flags], e.g. pixi run train_eval cuda:1 --soft-lambda 0.2

set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE=$1
shift

# 8 random hex characters, as a wandb run id
export WANDB_RUN_ID=$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')
echo "Run $WANDB_RUN_ID: train $* on $DEVICE, then evaluate"
pixi run train "$@" --device "$DEVICE"
pixi run evaluate --run "$WANDB_RUN_ID" --device "$DEVICE"
