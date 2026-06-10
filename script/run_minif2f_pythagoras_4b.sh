#!/usr/bin/env bash
set -euo pipefail

cd Pythagoras-Prover

python evaluation/minif2f.py \
  --model Pythagoras-LM//Pythagoras-Prover-4B \
  --output_dir outputs/minif2f_pythagoras_prover_4b \
  --n 16 \
  --tensor_parallel_size 8 \
  --debug \
  --trust_remote_code \
  "$@"
