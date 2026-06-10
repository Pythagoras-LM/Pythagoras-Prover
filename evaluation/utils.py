from __future__ import annotations

import re

from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

PROOF_SECTION_MARKER = "### Complete Lean 4 Proof"
LEAN_BLOCK_RE = re.compile(r"```lean(?:4)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
OPEN_LEAN_FENCE_RE = re.compile(r"```lean(?:4)?\s*\n", re.IGNORECASE)
TOP_LEVEL_LEAN_DECL_RE = re.compile(r"^\s*(theorem|lemma|example|def)\b", re.MULTILINE)
LEAN_DECL_PREFIXES = ("theorem ", "lemma ", "example", "def ")
DEFAULT_LEAN_HEADER = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
""".strip()
engines: dict[str, LLM] = {}


def formal_input(example: dict) -> str:
    return (example.get("formal_statement") or "").strip()


def generate_proofs(model_path, formal_statements, tokenizer, n=1, tensor_parallel_size=8, gpu_memory_utilization=0.90, trust_remote_code=False, seed=42, max_model_len=40960, temperature=1.0, top_p=0.95, max_tokens=40000):
    print(f"Using model: {model_path}")

    if model_path not in engines:
        engines[model_path] = LLM(model=model_path, tokenizer=model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=trust_remote_code, seed=seed, max_model_len=max_model_len)

    prompts = []
    for statement in formal_statements:
        prompt = PROMPT_TEMPLATE.format(statement.strip())
        messages = [{"role": "user", "content": prompt}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    params = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)

    print(f"Generating {n} proof(s) per problem...")
    outputs = engines[model_path].generate(prompts, params)

    solutions = []
    for output in outputs:
        proofs = [sample.text for sample in output.outputs]
        if n == 1:
            solutions.append(proofs[0] if proofs else "")
        else:
            solutions.append(proofs)

    return solutions


def build_records(examples: list[dict], proofs, n: int) -> list[dict]:
    records = []
    for example, proof_output in zip(examples, proofs):
        statement = formal_input(example)
        samples = []
        completions = proof_output if n > 1 else [proof_output]

        header_lines = []
        for line in statement.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("set_option ") or stripped.startswith("open "):
                header_lines.append(line)
                continue
            if stripped.startswith("/--") or stripped.startswith(LEAN_DECL_PREFIXES):
                break
        header = "\n".join(header_lines).strip() or DEFAULT_LEAN_HEADER

        for sample_id, completion in enumerate(completions):
            marker_index = completion.rfind(PROOF_SECTION_MARKER)
            if marker_index == -1:
                reasoning = completion.strip() or None
                complete_section = None
                search_text = completion
            else:
                reasoning = completion[:marker_index].strip() or None
                complete_section = completion[marker_index:].strip() or None
                search_text = completion[marker_index:]

            matches = LEAN_BLOCK_RE.findall(search_text)
            if matches:
                lean_source = matches[-1].strip() or None
            else:
                open_fences = list(OPEN_LEAN_FENCE_RE.finditer(search_text))
                lean_source = search_text[open_fences[-1].end() :].strip() if open_fences else None
                lean_source = lean_source or None
                stripped_completion = completion.strip()
                if lean_source is None and ((stripped_completion.startswith("import ") and TOP_LEVEL_LEAN_DECL_RE.search(stripped_completion)) or TOP_LEVEL_LEAN_DECL_RE.match(stripped_completion)):
                    lean_source = stripped_completion

            candidate_body = lean_source or completion
            if candidate_body.strip().startswith("import "):
                candidate = candidate_body.strip()
            else:
                candidate = f"{header}\n\n{candidate_body.strip()}".strip()

            verification = None
            if lean_source is not None:
                if lean_source.strip().startswith("import "):
                    verification = lean_source.strip()
                else:
                    verification = f"{header}\n\n{lean_source.strip()}".strip()

            samples.append({"sample_id": sample_id, "completion": completion, "reasoning": reasoning, "complete_lean_proof_section": complete_section, "lean_code_block": lean_source, "candidate_source": candidate, "verification_source": verification})

        records.append({"dataset_index": example["dataset_index"], "name": example.get("name"), "informal_prefix": example.get("informal_prefix"), "formal_statement": example.get("formal_statement"), "prompt_formal_statement": statement, "num_samples": n, "samples": samples})

    return records
