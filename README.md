<div align="center">
    <h1> <a href="https://pythagoras-prover.github.io/"> <strong>Pythagoras-Prover: Advancing Efficient Formal Proving via Augmented Lean Formalisation</strong></a></h1>
</div>

# Pythagoras-Prover
## 1. Introduction

We introduce Pythagoras-Prover, an open-source family of compute-efficient language models for automated formal proof generation in Lean 4. Our approach combines three key innovations: (1) Failure-mode-conditioned data synthesis: a rubric-guided distillation pipeline that re-targets each rejected seed against the specific Lean type-checker error responsible for its rejection, yielding a verified corpus partitioned into easy, medium and hard difficulty tiers and a 30% relative gain in autoformalisation success; (2) Curriculum-based parameter-efficient training: LoRA-only supervised fine-tuning over a three-stage difficulty curriculum under an 8K context with a dynamic proof-reasoning filter, followed by reinforcement learning with a binary Lean-compilation reward; (3) Augmented Lean Formalisation (ALF): a structured mutation scheme that expands each statement into roughly two million formal variants without per-instance Lean compilation, reused in a self-distillation stage.

Our small model, Pythagoras-Prover-4B, reaches 86.07% on MiniF2F-Test at Pass@32, surpassing the prior state-of-the-art DeepSeek-Prover-V2-671B (82.4%) while being roughly 167× smaller in parameter count, and scales to 89.75% at Pass@2048 under independent restart sampling. On the MiniF2F-ALF decontamination split, Pythagoras-Prover-4B reaches parity with Goedel-Prover-V2-32B at 8× fewer parameters. To our knowledge it is the smallest open-source model to surpass the 671B-parameter state of the art on MiniF2F-Test at Pass@32.

## 2. Benchmark Performance

<figure>
  <div class="fig-row">
    <div class="panel panel-1" style="width:100%;">
      <img src="https://github.com/Pythagoras-Prover-LM/Pythagoras-Prover/blob/main/assets/prover_fig1_hi.png" alt="…">
    </div>
  </div>
  <figcaption>
  <strong>Figure 1</strong>: <em>Pass@32 performance on MiniF2F, PutnamBench, and our new MiniF2F-ALF, by mutating MiniF2F-Statement through our proposed ALF method.</em>
  </figcaption>
</figure>


The figure above shows the state-of-the-art performance of Pythagoras-Prover. The results is reported all with Pass@32 unless specified. We observe that (1) Across all three datasets, Pythagoras-Prover-4B anmd 32B outperforms prior models, including Goedel-Prover-v2, DeepSeek-Prover-V2-671B and Kimina-Prover; (2) on miniF2F AND MiniF2F-ALF, our 4B model matches the performance of DeepSeek-Prover-V2-671B while being 167 times smaller in model size.

<div align="center">
  <table style="margin: 0 auto;">
    <thead>
      <tr>
        <th>#</th>
        <th>Model</th>
        <th>num-solved</th>
        <th>compute</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>1</td><td><strong>Pythagoras-Prover-32B</strong></td><td><strong>93</strong></td><td><strong>Pass@2048</strong></td></tr>
      <tr><td>1</td><td><strong>Pythagoras-Prover-32B</strong></td><td><strong>59</strong></td><td><strong>Pass@64</strong></td></tr>
      <tr><td>1</td><td><strong>Pythagoras-Prover-32B</strong></td><td><strong>48</strong></td><td><strong>Pass@32</strong></td></tr>
      <tr><td>2</td><td>Goedel-Prover-V2-32B (self-correction mode)</td><td>86</td><td>Pass@184</td></tr>
      <tr><td>2</td><td>Goedel-Prover-V2-32B (self-correction mode)</td><td>57</td><td>Pass@32</td></tr>
      <tr><td>2</td><td>Goedel-Prover-V2-32B</td><td>43</td><td>Pass@32</td></tr>
      <tr><td>3</td><td>DeepSeek-Prover-V2-671B</td><td>47</td><td>Pass@1024</td></tr>
      <tr><td>3</td><td>DeepSeek-Prover-V2-671B</td><td>22</td><td>Pass@32</td></tr>
      <tr><td>4</td><td>DSP+</td><td>23</td><td>Pass@128</td></tr>
      <tr><td>5</td><td>Bourbaki</td><td>14</td><td>Pass@512</td></tr>
      <tr><td>6</td><td>Kimina-Prover-7B-Distill</td><td>10</td><td>Pass@192</td></tr>
      <tr><td>7</td><td>Self-play Theorem Prover</td><td>8</td><td>Pass@3200</td></tr>
      <tr><td>8</td><td>Goedel-Prover-SFT</td><td>7</td><td>Pass@512</td></tr>
      <tr><td>9</td><td>ABEL (closed-source)</td><td>7</td><td>Pass@596</td></tr>
    </tbody>
  </table>
  <!-- table caption -->
  <caption align="bottom"><strong>Table 1</strong>: <em>PutnamBench leaderboard (problems solved out of 672). Pythagoras-Prover-32B secures the top rank with 93 problems at Pass@2048, surpassing the previous best (Goedel-Prover-V2-32B, 86 at Pass@184 with self-correction) by 7 problems and roughly doubling DeepSeek-Prover-V2-671B's pass@1024 result, despite being roughly 20× smaller. We omit Seed-Prover (331 solved; closed-source with undisclosed test-time compute) from the ranked rows.</em></caption>
</div>


## 3. Model & Dataset Download
We release Pythagoras-Prover in two model sizes: 4B and 32B parameters. Pythagoras-Prover is trained based on Qwen3 model series. We release our Pythagoras-Prover models, datasets and our new benchmark for future research. 

<div align="center">
  
| Model | Download |
| -------- | -------- |
|    Pythagoras-Prover-32B    |   Coming Soon   |
|    Pythagoras-Prover-4B    |   [🤗Download](https://huggingface.co/Pythagoras-LM/
Pythagoras-Prover-4B)    |
|    Pythagoras-Prover-Diffusion-4B    |   Coming Soon   |

</div>

<div align="center">

| Dataset | Download |
| -------- | -------- |
|   Pythagoras-Prover-SFT    |   Coming Soon    |
|   Pythagoras-Prover-Distill-4B    |   Coming Soon    |
|    Pythagoras-Prover-Distill-32B    |   Coming Soon    |
</div>

## 4. Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Pythagoras-LM/Pythagoras-Prover-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

formal_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The volume of a cone is given by the formula $V = \frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.-/
theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))
(h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by
  sorry
""".strip()

prompt = """
Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

chat = [
  {"role": "user", "content": prompt.format(formal_statement)},
]

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

import time
start = time.time()
outputs = model.generate(inputs, max_new_tokens=8192)
print(tokenizer.batch_decode(outputs))
print(time.time() - start)
```

# 7. Cite
COMING SOON