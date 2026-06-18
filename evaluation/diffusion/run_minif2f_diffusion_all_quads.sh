#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for quad in 1 2 3 4; do
  QUAD="${quad}" "${SCRIPT_DIR}/run_minif2f_diffusion.sh" "$@"
done
