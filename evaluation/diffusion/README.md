# Diffusion MiniF2F

This folder vendors the `dllm` source needed by `minif2f.py`, so it does not depend on another local checkout being present. The vendored `dllm` code is from [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm), associated with the [dLLM paper](https://arxiv.org/abs/2602.22661). The Apache-2.0 license text is included in `LICENSE`.

Install the runtime dependencies from `requirements.txt`, then run with an A2D BD3LM checkpoint:

```bash
MODEL_PATH=/path/to/a2d_bd3lm_checkpoint ./diffusion/run_minif2f_diffusion.sh
```

To run all four shards:

```bash
MODEL_PATH=/path/to/a2d_bd3lm_checkpoint ./diffusion/run_minif2f_diffusion_all_quads.sh
```

Outputs are written to `outputs/diffusion_minif2f_bd3lm_4b_pass32_quadXX/` unless `OUTPUT_DIR` or `OUTPUT_ROOT` is set.
