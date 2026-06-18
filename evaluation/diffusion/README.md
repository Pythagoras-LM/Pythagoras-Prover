# Diffusion MiniF2F

This folder vendors the `dllm` source needed by `minif2f.py`, so it does not depend on another local checkout being present. The vendored `dllm` code is from [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm), associated with the [dLLM paper](https://arxiv.org/abs/2602.22661). The Apache-2.0 license text is included in `LICENSE`.

## Setup

Install runtime dependencies (Python 3.10+ recommended):

```bash
pip install -r evaluation/diffusion/requirements.txt
```


## Running

### Single shard (default: quad 1 of 4, 8 GPUs)

```bash
MODEL_PATH=/path/to/a2d_bd3lm_checkpoint \
  ./evaluation/diffusion/run_minif2f_diffusion.sh
```

### All four shards sequentially

```bash
MODEL_PATH=/path/to/a2d_bd3lm_checkpoint \
  ./evaluation/diffusion/run_minif2f_diffusion_all_quads.sh
```

### Environment variables accepted by the shell scripts

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | *(required)* | Path to A2D BD3LM checkpoint |
| `OUTPUT_DIR` | `outputs/diffusion_minif2f_bd3lm_4b_pass32_quad<N>` | Output directory |
| `OUTPUT_ROOT` | `outputs/diffusion_minif2f_bd3lm_4b_pass32` | Root when `OUTPUT_DIR` is not set |
| `N` | `32` | Samples per problem (pass@k budget) |
| `MAX_TOKENS` | `8096` | Max generated tokens per sample |
| `BLOCK_SIZE` | `32` | BD3LM denoising block size |
| `SAMPLE_BATCH_SIZE` | `4` | Problems processed per GPU batch |
| `SPLIT` | `4` | Total number of shards |
| `QUAD` | `1` | 1-based shard index to evaluate |
| `NUM_PROCESSES` | `8` | Number of GPU processes for `accelerate launch` |

### Running a single problem subset directly

```bash
cd /path/to/Pythagoras-Prover
PYTHONPATH=evaluation/diffusion python evaluation/diffusion/minif2f.py \
  --model /path/to/checkpoint \
  --n 4 \
  --max_tokens 4096 \
  --output_dir outputs/debug_run
```

Pass `--resume_existing` to continue an interrupted run without regenerating completed records.

## Output format

Each shard writes to `<OUTPUT_DIR>/minif2f_test_quad<N>-of-4_samples.jsonl` (one JSON object per line, one line per problem) and a companion `_summary.json`. Each JSONL record has:

```jsonc
{
  "dataset_index": 0,
  "name": "mathd_algebra_...",
  "formal_statement": "...",
  "num_samples": 32,
  "samples": [
    {
      "sample_id": 0,
      "completion": "<full model output>",
      "reasoning": "<text before ### Complete Lean 4 Proof>",
      "lean_code_block": "<extracted lean4 code block>",
      "candidate_source": "<lean_code_block with Mathlib imports prepended>",
      "verification_source": "<same, or null if no lean block found>"
    }
    // ... 31 more
  ],
  "generation_metrics": { /* per-problem throughput / FLOP counters */ }
}
```

The `verification_source` field is what should be fed to Lean / `mathlib4` for proof checking.

