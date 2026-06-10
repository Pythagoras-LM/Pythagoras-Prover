#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  verify.sh INPUT_FILE [OUTPUT_FILE] [N_CPUS] [TIMEOUT] [STATEMENT_MATCH]

Defaults:
  OUTPUT_FILE      input path with _verified.json suffix
  N_CPUS           16
  TIMEOUT          120
  STATEMENT_MATCH  exact

STATEMENT_MATCH:
  exact       DeepSeek-compatible evaluation: mismatched target statement is failed.
  substitute  Replace generated theorem declaration with the input formal_statement before verification.

Example:
  ./verify.sh /path/to/minif2f_test_part01-of-08_samples.jsonl
  ./verify.sh /path/to/samples.jsonl /path/to/samples_verified.json 16 120 exact
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/lustre/scratch/users/joshua.ong/envs/diffu-sft/bin/python}"
INPUT_FILE="$1"
OUTPUT_FILE="${2:-}"
N_CPUS="${3:-16}"
TIMEOUT="${4:-120}"
STATEMENT_MATCH="${5:-exact}"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input file not found: $INPUT_FILE" >&2
  exit 2
fi

if [[ "$STATEMENT_MATCH" != "exact" && "$STATEMENT_MATCH" != "substitute" ]]; then
  echo "STATEMENT_MATCH must be 'exact' or 'substitute', got: $STATEMENT_MATCH" >&2
  exit 2
fi

cmd=("$PYTHON_BIN" "$REPO_DIR/lean_compiler/verify_formal_proof.py" --input_file "$INPUT_FILE" --n_cpus "$N_CPUS" --timeout "$TIMEOUT" --statement_match "$STATEMENT_MATCH")
if [[ -n "$OUTPUT_FILE" ]]; then
  cmd+=(--output_file "$OUTPUT_FILE")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
