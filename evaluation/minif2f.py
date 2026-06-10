from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer

from utils import build_records, formal_input, generate_proofs


INDEX_COLUMN = "_minif2f_original_dataset_index"


def parse_args():
    parser = argparse.ArgumentParser(description="MiniF2F evaluation with vLLM")
    parser.add_argument("--model", type=str, default="Pythagoras-LM/Pythagoras-Prover-4B")
    parser.add_argument("--dataset_name", type=str, default="AI-MO/minif2f_test")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="outputs/minif2f_pythagoras_prover_4b")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def load_minif2f(args):
    try:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    except ValueError as exc:
        if "Unknown split" not in str(exc):
            raise
        dataset = load_dataset(args.dataset_name, split="train")
        if args.dataset_split != "train" and "split" in dataset.column_names:
            dataset = dataset.filter(lambda ex: ex["split"] == args.dataset_split)

    dataset = dataset.add_column(INDEX_COLUMN, list(range(len(dataset))))
    if args.debug:
        dataset = dataset.select(range(len(dataset) // 2))

    examples = []
    for row in dataset:
        example = dict(row)
        example["dataset_index"] = int(example.pop(INDEX_COLUMN))
        examples.append(example)

    return examples


def output_path(args):
    return os.path.join(args.output_dir, f"minif2f_{args.dataset_split}_samples.jsonl")


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    examples = load_minif2f(args)
    print(f"Loaded {len(examples)} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    statements = [formal_input(example) for example in examples]
    proofs = generate_proofs(args.model, statements, tokenizer, n=args.n, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=args.trust_remote_code, seed=args.seed, max_model_len=args.max_model_len, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

    records = build_records(examples, proofs, args.n)
    samples_path = output_path(args)
    write_jsonl(samples_path, records)

    print(f"[OK] Wrote {len(records)} records to: {samples_path}")


if __name__ == "__main__":
    main()
