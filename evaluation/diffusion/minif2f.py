"""Run MiniF2F sampling with an A2D BD3LM checkpoint."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from tqdm.auto import tqdm

os.environ.setdefault(
    "MPLCONFIGDIR",
    f"/tmp/{os.environ.get('USER') or getpass.getuser()}/matplotlib_minif2f",
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import dllm


PROMPT_TEMPLATE = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

LEAN_BLOCK_RE = re.compile(r"```lean(?:4)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
OPEN_LEAN_FENCE_RE = re.compile(r"```lean(?:4)?\s*\n", re.IGNORECASE)
PROOF_SECTION_MARKER = "### Complete Lean 4 Proof"
TOP_LEVEL_LEAN_DECL_RE = re.compile(r"^\s*(theorem|lemma|example|def)\b", re.MULTILINE)
DEFAULT_LEAN_HEADER = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
""".strip()


@dataclass(frozen=True)
class ProcessInfo:
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def get_process_info() -> ProcessInfo:
    """Read process placement from accelerate/torchrun env without initializing NCCL."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            local_rank = local_rank % device_count
            torch.cuda.set_device(local_rank)

    return ProcessInfo(rank=rank, world_size=world_size, local_rank=local_rank)


def build_formal_input(example: dict) -> str:
    return (example.get("formal_statement") or "").strip()


def extract_reasoning(text: str) -> str | None:
    marker_index = text.rfind(PROOF_SECTION_MARKER)
    if marker_index == -1:
        stripped = text.strip()
        return stripped or None
    reasoning = text[:marker_index].strip()
    return reasoning or None


def extract_complete_lean_proof_section(text: str) -> str | None:
    marker_index = text.rfind(PROOF_SECTION_MARKER)
    if marker_index == -1:
        return None
    section = text[marker_index:].strip()
    return section or None


def extract_last_lean_block(text: str) -> str | None:
    marker_index = text.rfind(PROOF_SECTION_MARKER)
    search_text = text[marker_index:] if marker_index != -1 else text
    matches = LEAN_BLOCK_RE.findall(search_text)
    if matches:
        return matches[-1].strip() or None

    open_fences = list(OPEN_LEAN_FENCE_RE.finditer(search_text))
    if not open_fences:
        return None

    trailing_block = search_text[open_fences[-1].end() :].strip()
    return trailing_block or None


def looks_like_full_lean_source(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("import ") and TOP_LEVEL_LEAN_DECL_RE.search(stripped):
        return True
    return TOP_LEVEL_LEAN_DECL_RE.match(stripped) is not None


def extract_lean_source(text: str) -> str | None:
    lean_block = extract_last_lean_block(text)
    if lean_block is not None:
        return lean_block
    if looks_like_full_lean_source(text):
        return text.strip()
    return None


def extract_header_from_formal_statement(formal_statement: str) -> str:
    header_lines = []
    for line in formal_statement.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("set_option ")
            or stripped.startswith("open ")
        ):
            header_lines.append(line)
            continue
        if stripped.startswith("/--") or TOP_LEVEL_LEAN_DECL_RE.match(stripped):
            break
    return "\n".join(header_lines).strip() or DEFAULT_LEAN_HEADER


def ensure_lean_header(source: str, formal_statement: str = "") -> str:
    stripped = source.strip()
    if stripped.startswith("import "):
        return stripped
    header = extract_header_from_formal_statement(formal_statement)
    return f"{header}\n\n{stripped}".strip()


def assemble_candidate_source(formal_input: str, completion: str) -> str:
    lean_block = extract_lean_source(completion)
    if lean_block is not None:
        return ensure_lean_header(lean_block, formal_input)
    return ensure_lean_header(completion, formal_input)


def build_verification_source(formal_statement: str, completion: str) -> str | None:
    lean_block = extract_lean_source(completion)
    if lean_block is None:
        return None
    return ensure_lean_header(lean_block, formal_statement)


def finite_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def clean_left_padding(tokenizer, seq_ids: list[int]) -> list[int]:
    full = list(seq_ids)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        while full and full[0] == pad_id:
            full.pop(0)
    return full


def split_generated_token_ids(
    tokenizer, seq_ids: list[int], input_ids: list[int]
) -> tuple[list[int], list[int]]:
    full = clean_left_padding(tokenizer, seq_ids)
    start = len(input_ids)
    canvas_ids = full[start:]
    end = len(full)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    eot_id = getattr(tokenizer, "eot_token_id", None)
    stop_ids = {token_id for token_id in (eos_id, eot_id) if token_id is not None}
    if stop_ids:
        for index in range(start, len(full)):
            if full[index] in stop_ids:
                end = index
                break

    return canvas_ids, full[start:end]


def empty_generation_metrics() -> dict:
    return {
        "wall_seconds": 0.0,
        "sample_batches": 0,
        "samples": 0,
        "instances": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "canvas_tokens": 0,
        "max_new_tokens_budget": 0,
        "model_forward_calls": 0,
        "prefix_forward_calls": 0,
        "denoise_forward_calls": 0,
        "other_forward_calls": 0,
        "forward_query_tokens": 0,
        "forward_kv_tokens": 0,
        "prefix_query_tokens": 0,
        "denoise_query_tokens": 0,
        "estimated_transformer_flops": 0.0,
        "estimated_lm_head_flops": 0.0,
        "estimated_total_flops": 0.0,
    }


def finalize_generation_metrics(metrics: dict, cfg_scale: float) -> dict:
    finalized = dict(metrics)
    cfg_multiplier = 2 if cfg_scale > 0.0 else 1
    diffusion_steps = finalized["denoise_forward_calls"] / cfg_multiplier
    prefix_blocks = finalized["prefix_forward_calls"] / cfg_multiplier
    wall_seconds = finalized["wall_seconds"]
    output_tokens = finalized["output_tokens"]
    canvas_tokens = finalized["canvas_tokens"]
    total_flops = finalized["estimated_total_flops"]
    transformer_flops = finalized["estimated_transformer_flops"]

    finalized.update(
        {
            "cfg_forward_multiplier": cfg_multiplier,
            "observed_diffusion_steps": diffusion_steps,
            "observed_prefix_blocks": prefix_blocks,
            "output_tokens_per_second": finite_divide(output_tokens, wall_seconds),
            "canvas_tokens_per_second": finite_divide(canvas_tokens, wall_seconds),
            "samples_per_second": finite_divide(finalized["samples"], wall_seconds),
            "instances_per_second": finite_divide(finalized["instances"], wall_seconds),
            "output_tokens_per_diffusion_step": finite_divide(
                output_tokens, diffusion_steps
            ),
            "canvas_tokens_per_diffusion_step": finite_divide(
                canvas_tokens, diffusion_steps
            ),
            "forward_query_tokens_per_second": finite_divide(
                finalized["forward_query_tokens"], wall_seconds
            ),
            "estimated_total_tflops_per_second": finite_divide(
                total_flops / 1e12, wall_seconds
            ),
            "estimated_transformer_tflops_per_second": finite_divide(
                transformer_flops / 1e12, wall_seconds
            ),
            "estimated_total_flops_per_output_token": finite_divide(
                total_flops, output_tokens
            ),
            "estimated_total_flops_per_canvas_token": finite_divide(
                total_flops, canvas_tokens
            ),
            "estimated_total_flops_per_instance": finite_divide(
                total_flops, finalized["instances"]
            ),
        }
    )
    return finalized


def add_generation_metrics(total: dict, update: dict) -> None:
    for key in empty_generation_metrics():
        total[key] += update.get(key, 0)


def estimate_forward_flops(config, batch_size: int, query_len: int, kv_len: int) -> dict:
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    intermediate_size = int(getattr(config, "intermediate_size", 0) or 0)
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads) or num_heads)
    vocab_size = int(getattr(config, "vocab_size", 0) or 0)

    if not hidden_size or not intermediate_size or not num_layers or not num_heads:
        return {
            "transformer_flops": 0.0,
            "lm_head_flops": 0.0,
            "total_flops": 0.0,
        }

    head_dim = hidden_size // num_heads
    kv_hidden_size = num_kv_heads * head_dim
    batch_query = batch_size * query_len

    qkv_o_flops = (
        2 * batch_query * hidden_size * hidden_size
        + 2 * batch_query * hidden_size * kv_hidden_size
        + 2 * batch_query * hidden_size * kv_hidden_size
        + 2 * batch_query * hidden_size * hidden_size
    )
    attention_flops = 4 * batch_size * query_len * kv_len * hidden_size
    mlp_flops = 6 * batch_query * hidden_size * intermediate_size
    transformer_flops = num_layers * (qkv_o_flops + attention_flops + mlp_flops)
    lm_head_flops = 2 * batch_query * hidden_size * vocab_size

    return {
        "transformer_flops": float(transformer_flops),
        "lm_head_flops": float(lm_head_flops),
        "total_flops": float(transformer_flops + lm_head_flops),
    }


@contextmanager
def capture_model_forward_metrics(model):
    stats = empty_generation_metrics()
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]

        if torch.is_tensor(input_ids) and input_ids.ndim >= 2:
            batch_size = int(input_ids.shape[0])
            query_len = int(input_ids.shape[1])
            attention_mask = kwargs.get("attention_mask")
            if torch.is_tensor(attention_mask) and attention_mask.ndim >= 2:
                kv_len = int(attention_mask.shape[-1])
            else:
                kv_len = query_len

            past_key_values = kwargs.get("past_key_values")
            use_cache = kwargs.get("use_cache")
            category = "other"
            if past_key_values is not None:
                category = "denoise"
            elif use_cache:
                category = "prefix"

            flops = estimate_forward_flops(
                model.config,
                batch_size=batch_size,
                query_len=query_len,
                kv_len=kv_len,
            )
            query_tokens = batch_size * query_len
            kv_tokens = batch_size * kv_len

            stats["model_forward_calls"] += 1
            stats["forward_query_tokens"] += query_tokens
            stats["forward_kv_tokens"] += kv_tokens
            stats["estimated_transformer_flops"] += flops["transformer_flops"]
            stats["estimated_lm_head_flops"] += flops["lm_head_flops"]
            stats["estimated_total_flops"] += flops["total_flops"]
            stats[f"{category}_forward_calls"] += 1
            if category in ("prefix", "denoise"):
                stats[f"{category}_query_tokens"] += query_tokens
        return original_forward(*args, **kwargs)

    model.forward = wrapped_forward
    try:
        yield stats
    finally:
        model.forward = original_forward


def summarize_generation_metrics(output_path: Path, expected_n: int, cfg_scale: float) -> dict:
    aggregate = empty_generation_metrics()
    records_with_metrics = 0
    for record in iter_complete_jsonl(output_path, expected_n):
        metrics = record.get("generation_metrics")
        if not isinstance(metrics, dict):
            continue
        add_generation_metrics(aggregate, metrics)
        records_with_metrics += 1

    finalized = finalize_generation_metrics(aggregate, cfg_scale=cfg_scale)
    finalized["records_with_metrics"] = records_with_metrics
    return finalized


def load_completed_dataset_indices(output_path: Path, expected_n: int) -> set[int]:
    completed: set[int] = set()
    if not output_path.is_file():
        return completed

    with output_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Skipping incomplete JSONL line while resuming: {output_path}:{line_no}",
                    flush=True,
                )
                continue

            dataset_index = record.get("dataset_index")
            samples = record.get("samples") or []
            if (
                isinstance(dataset_index, int)
                and record.get("num_samples") == expected_n
                and len(samples) == expected_n
            ):
                completed.add(dataset_index)

    return completed


def iter_complete_jsonl(path: Path, expected_n: int):
    if not path.is_file():
        return

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping incomplete JSONL line: {path}:{line_no}", flush=True)
                continue

            samples = record.get("samples") or []
            if (
                isinstance(record.get("dataset_index"), int)
                and record.get("num_samples") == expected_n
                and len(samples) == expected_n
            ):
                yield record


def rank_output_path(output_path: Path, process_index: int) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.rank{process_index:05d}{output_path.suffix}"
    )


def count_complete_jsonl(path: Path, expected_n: int) -> int:
    if not path.is_file():
        return 0
    return sum(1 for _ in iter_complete_jsonl(path, expected_n))


def wait_for_rank_record_counts(
    rank_paths: list[Path],
    target_counts: list[int],
    expected_n: int,
    poll_seconds: int = 30,
) -> None:
    while True:
        counts = [count_complete_jsonl(path, expected_n) for path in rank_paths]
        missing = [
            f"rank{rank}: {counts[rank]}/{target}"
            for rank, target in enumerate(target_counts)
            if counts[rank] < target
        ]
        if not missing:
            return

        print(
            "Waiting for rank-local outputs before merge: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else ""),
            flush=True,
        )
        time.sleep(poll_seconds)


def merge_rank_outputs(
    output_path: Path,
    rank_paths: list[Path],
    expected_n: int,
    resume_existing: bool,
) -> int:
    records_by_index: dict[int, dict] = {}

    if resume_existing:
        for record in iter_complete_jsonl(output_path, expected_n):
            records_by_index[record["dataset_index"]] = record

    for path in rank_paths:
        for record in iter_complete_jsonl(path, expected_n):
            records_by_index[record["dataset_index"]] = record

    records = sorted(records_by_index.values(), key=lambda record: record["dataset_index"])
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return len(records)


def shard_bounds(total: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    if num_shards < 1:
        raise ValueError(f"--split must be >= 1, got {num_shards}")
    if not 1 <= shard_index <= num_shards:
        raise ValueError(f"--quad must be in [1, {num_shards}], got {shard_index}")

    start = total * (shard_index - 1) // num_shards
    end = total * shard_index // num_shards
    return start, end


def load_minif2f_dataset(args: argparse.Namespace):
    try:
        dataset = load_dataset(
            args.dataset_name,
            split=args.dataset_split,
            trust_remote_code=args.trust_remote_code,
        )
        loaded_split = args.dataset_split
    except ValueError as exc:
        if "Unknown split" not in str(exc):
            raise

        dataset = load_dataset(
            args.dataset_name,
            split="train",
            trust_remote_code=args.trust_remote_code,
        )
        loaded_split = "train"

        if args.dataset_split != loaded_split and "split" in dataset.column_names:
            dataset = dataset.filter(
                lambda ex: ex["split"] == args.dataset_split,
                desc=f"Filtering logical split={args.dataset_split}",
            )

    original_len = len(dataset)
    original_index_column = "_minif2f_original_dataset_index"
    if original_index_column in dataset.column_names:
        raise ValueError(f"Dataset already contains reserved column: {original_index_column}")
    dataset = dataset.add_column(original_index_column, list(range(original_len)))

    selection_limit = original_len
    if args.half:
        selection_limit = min(selection_limit, original_len // 2)
    if args.quarter:
        selection_limit = min(selection_limit, original_len // 4)
    if args.max_examples is not None:
        selection_limit = min(selection_limit, args.max_examples)
    if selection_limit < original_len:
        print(f"Selecting first {selection_limit} of {original_len} examples")
        dataset = dataset.select(range(selection_limit))

    selected_len = len(dataset)
    shard_start, shard_end = shard_bounds(selected_len, args.num_shards, args.shard_index)
    if args.num_shards > 1:
        if shard_start == shard_end:
            print(
                f"Selecting quad {args.shard_index}/{args.num_shards}: "
                f"empty shard at row {shard_start} of {selected_len}"
            )
        else:
            print(
                f"Selecting quad {args.shard_index}/{args.num_shards}: "
                f"rows {shard_start}-{shard_end - 1} of {selected_len}"
            )
        dataset = dataset.select(range(shard_start, shard_end))

    return dataset, loaded_split


def configure_hf_hub_timeouts(args: argparse.Namespace) -> None:
    etag_timeout = max(1, int(args.hf_hub_etag_timeout))
    download_timeout = max(1, int(args.hf_hub_download_timeout))
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(etag_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(download_timeout)
    print(
        "Hugging Face Hub timeouts: "
        f"etag={etag_timeout}s, download={download_timeout}s"
    )


def configure_rank_local_hf_cache(rank: int) -> None:
    """Keep launcher ranks away from shared HF dataset file locks."""
    if os.environ.get("HF_DATASETS_CACHE"):
        return

    user = os.environ.get("USER") or getpass.getuser()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    cache_root = f"/tmp/{user}/hf_minif2f_{job_id}_rank{rank}"
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets")
    os.environ.setdefault("HF_MODULES_CACHE", os.path.join(cache_root, "modules"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_root, "hub"))
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)


def apply_chat_template(tokenizer, template_name: str) -> None:
    if template_name == "auto":
        if tokenizer.chat_template is None:
            raise ValueError("Tokenizer has no embedded chat template. Pass --template qwen3_nothink.")
        return

    if template_name != "qwen3_nothink":
        raise ValueError(f"Unsupported template: {template_name}")

    if tokenizer.eos_token != "<|im_end|>":
        tokenizer.eos_token = "<|im_end|>"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    original_apply_chat_template = tokenizer.apply_chat_template

    def apply_chat_template_no_think(*args, **kwargs):
        if "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = False
        try:
            return original_apply_chat_template(*args, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return original_apply_chat_template(*args, **kwargs)

    tokenizer.apply_chat_template = apply_chat_template_no_think


def load_model_for_process(
    model_args: dllm.utils.ModelArguments, process: ProcessInfo
) -> transformers.PreTrainedModel:
    """Load one full model copy on this process's local GPU without distributed state."""
    device_map = (
        {"": process.local_rank}
        if torch.cuda.is_available()
        and not transformers.modeling_utils.is_deepspeed_zero3_enabled()
        else None
    )

    quant_config = None
    if model_args.load_in_4bit and transformers.utils.is_bitsandbytes_available():
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_args.dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    params = {
        "dtype": model_args.dtype,
        "device_map": device_map,
        "quantization_config": quant_config,
        "attn_implementation": model_args.attn_implementation,
    }

    try:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path, **params
        )
    except Exception:
        model = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path, **params
        )

    if model_args.load_in_4bit and quant_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )

    return dllm.utils.load_peft(model, model_args)


def encode_prompt(tokenizer, prompt_text: str) -> list[int]:
    messages = [{"role": "user", "content": prompt_text}]
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        input_ids = tokenizer.apply_chat_template(messages, **kwargs)

    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def build_prompt_records(dataset, tokenizer) -> tuple[list[dict], list[list[int]]]:
    examples: list[dict] = []
    prompts: list[list[int]] = []
    print("Building MiniF2F prompts...")

    for dataset_idx in tqdm(range(len(dataset)), desc="Prompts"):
        example = dict(dataset[dataset_idx])
        original_dataset_index = example.pop(
            "_minif2f_original_dataset_index", dataset_idx
        )
        example["_dataset_index"] = int(original_dataset_index)

        formal_input = build_formal_input(example)
        prompt_text = PROMPT_TEMPLATE.format(formal_input)
        prompt_ids = encode_prompt(tokenizer, prompt_text)

        examples.append(example)
        prompts.append(prompt_ids)

    return examples, prompts


def build_sampler_config(args: argparse.Namespace) -> dllm.core.samplers.BD3LMSamplerConfig:
    steps = args.steps if args.steps is not None else args.max_new_tokens
    return dllm.core.samplers.BD3LMSamplerConfig(
        max_new_tokens=args.max_new_tokens,
        steps=steps,
        steps_per_block=args.steps_per_block,
        block_size=args.block_size,
        temperature=args.temperature,
        remasking=args.remasking,
        stochastic_transfer=args.stochastic_transfer,
        cfg_scale=args.cfg_scale,
        right_shift_logits=args.right_shift_logits,
        return_dict=False,
    )


def validate_a2d_qwen3_model(model) -> None:
    expected_config_cls = dllm.pipelines.a2d.A2DQwen3Config
    expected_model_cls = dllm.pipelines.a2d.A2DQwen3LMHeadModel
    if not isinstance(model.config, expected_config_cls):
        raise TypeError(
            "Expected A2DQwen3Config for this MiniF2F eval, got "
            f"{type(model.config).__module__}.{type(model.config).__name__} "
            f"(model_type={getattr(model.config, 'model_type', None)!r})."
        )
    if not isinstance(model, expected_model_cls):
        raise TypeError(
            "Expected A2DQwen3LMHeadModel for this MiniF2F eval, got "
            f"{type(model).__module__}.{type(model).__name__}."
        )


def infer_steps_per_block(
    max_new_tokens: int,
    steps: int,
    block_size: int,
    explicit_steps_per_block: int | None,
) -> tuple[int, int]:
    num_blocks = math.ceil(max_new_tokens / block_size)
    if explicit_steps_per_block is not None:
        return num_blocks, explicit_steps_per_block
    return num_blocks, math.ceil(steps / num_blocks)


def generate_completions(
    sampler: dllm.core.samplers.BD3LMSampler,
    tokenizer,
    sampler_config: dllm.core.samplers.BD3LMSamplerConfig,
    prompt_ids: list[int],
    num_samples: int,
    sample_batch_size: int,
    empty_cache: bool,
    show_progress: bool,
) -> tuple[list[str], dict]:
    completions: list[str] = []
    aggregate_metrics = empty_generation_metrics()
    batch_metrics: list[dict] = []

    for start in tqdm(
        range(0, num_samples, sample_batch_size),
        desc="Samples",
        leave=False,
        disable=not show_progress,
    ):
        cur_batch = min(sample_batch_size, num_samples - start)
        batch_inputs = [list(prompt_ids) for _ in range(cur_batch)]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with capture_model_forward_metrics(sampler.model) as forward_metrics:
            generated_ids = sampler.sample(
                inputs=batch_inputs,
                config=sampler_config,
                return_dict=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - start_time

        sequence_ids = generated_ids.tolist()
        output_token_ids = []
        canvas_token_ids = []
        for sequence, input_ids in zip(sequence_ids, batch_inputs):
            canvas_ids, trimmed_ids = split_generated_token_ids(
                tokenizer, sequence, input_ids
            )
            canvas_token_ids.append(canvas_ids)
            output_token_ids.append(trimmed_ids)

        completions.extend(
            tokenizer.decode(ids, skip_special_tokens=True) for ids in output_token_ids
        )

        metrics = dict(forward_metrics)
        metrics["wall_seconds"] = wall_seconds
        metrics["sample_batches"] = 1
        metrics["samples"] = cur_batch
        metrics["prompt_tokens"] = sum(len(input_ids) for input_ids in batch_inputs)
        metrics["output_tokens"] = sum(len(ids) for ids in output_token_ids)
        metrics["canvas_tokens"] = sum(len(ids) for ids in canvas_token_ids)
        metrics["max_new_tokens_budget"] = cur_batch * sampler_config.max_new_tokens
        finalized_batch_metrics = finalize_generation_metrics(
            metrics, cfg_scale=sampler_config.cfg_scale
        )
        finalized_batch_metrics["sample_start"] = start
        finalized_batch_metrics["sample_end"] = start + cur_batch
        batch_metrics.append(finalized_batch_metrics)
        add_generation_metrics(aggregate_metrics, metrics)

        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate_metrics["instances"] = 1
    finalized_metrics = finalize_generation_metrics(
        aggregate_metrics, cfg_scale=sampler_config.cfg_scale
    )
    finalized_metrics["batch_metrics"] = batch_metrics
    return completions, finalized_metrics


def make_record(example: dict, completions: list[str], generation_metrics: dict) -> dict:
    formal_input = build_formal_input(example)
    samples = []
    for sample_id, completion in enumerate(completions):
        samples.append(
            {
                "sample_id": sample_id,
                "completion": completion,
                "reasoning": extract_reasoning(completion),
                "complete_lean_proof_section": extract_complete_lean_proof_section(
                    completion
                ),
                "lean_code_block": extract_lean_source(completion),
                "candidate_source": assemble_candidate_source(formal_input, completion),
                "verification_source": build_verification_source(
                    example.get("formal_statement", ""),
                    completion,
                ),
            }
        )

    return {
        "dataset_index": example["_dataset_index"],
        "name": example.get("name"),
        "informal_prefix": example.get("informal_prefix"),
        "formal_statement": example.get("formal_statement"),
        "prompt_formal_statement": formal_input,
        "num_samples": len(completions),
        "samples": samples,
        "generation_metrics": generation_metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniF2F sampling with A2D BD3LM.")
    parser.add_argument(
        "--model",
        "--model_name_or_path",
        dest="model_name_or_path",
        type=str,
        required=True,
        help="A2D BD3LM checkpoint path.",
    )
    parser.add_argument("--dataset_name", type=str, default="AI-MO/minif2f_test")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/diffusion_minif2f_bd3lm_4b_pass32",
    )
    parser.add_argument("--n", type=int, default=32, help="Number of samples per problem.")
    parser.add_argument(
        "--max_tokens",
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=8192,
        help="Maximum generated tokens per sample.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="BD3LM denoising steps. Defaults to --max_tokens.",
    )
    parser.add_argument("--steps_per_block", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--remasking",
        type=str,
        default="low_confidence",
        choices=["low_confidence", "random"],
    )
    parser.add_argument("--stochastic_transfer", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--right_shift_logits", action="store_true")
    parser.add_argument("--sample_batch_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--quarter", action="store_true")
    parser.add_argument(
        "--num_shards",
        "--split",
        dest="num_shards",
        type=int,
        default=1,
        help="Split selected MiniF2F examples into this many contiguous shards.",
    )
    parser.add_argument(
        "--shard_index",
        "--quad",
        dest="shard_index",
        type=int,
        default=1,
        help="1-based shard index to evaluate.",
    )
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--template", type=str, default="qwen3_nothink", choices=["qwen3_nothink", "auto"])
    parser.add_argument("--hf_hub_etag_timeout", type=int, default=60)
    parser.add_argument("--hf_hub_download_timeout", type=int, default=120)
    parser.add_argument(
        "--distributed_timeout_seconds",
        type=int,
        default=21600,
        help="Kept for command compatibility; MiniF2F eval does not use NCCL barriers.",
    )
    parser.add_argument("--empty_cache", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.sample_batch_size < 1:
        raise ValueError("--sample_batch_size must be >= 1")

    process = get_process_info()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = (
        f"_quad{args.shard_index:02d}-of-{args.num_shards:02d}"
        if args.num_shards > 1
        else ""
    )
    output_path = output_dir / f"minif2f_{args.dataset_split}{shard_suffix}_samples.jsonl"
    summary_path = output_dir / f"minif2f_{args.dataset_split}{shard_suffix}_summary.json"
    rank_paths = [
        rank_output_path(output_path, rank)
        for rank in range(process.world_size)
    ]
    local_output_path = (
        rank_paths[process.rank] if process.world_size > 1 else output_path
    )
    if process.world_size > 1 and not args.resume_existing and local_output_path.exists():
        local_output_path.unlink()

    configure_hf_hub_timeouts(args)
    configure_rank_local_hf_cache(process.rank)
    transformers.set_seed(args.seed)

    dataset, loaded_split = load_minif2f_dataset(args)
    print(f"Loaded {len(dataset)} examples (split={args.dataset_split}, hf_split={loaded_split})")

    print(f"Loading model: {args.model_name_or_path}")
    model_args = dllm.utils.ModelArguments(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        attn_implementation=args.attn_implementation,
    )
    model = load_model_for_process(model_args=model_args, process=process).eval()
    validate_a2d_qwen3_model(model)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    apply_chat_template(tokenizer, args.template)

    sampler = dllm.core.samplers.BD3LMSampler(model=model, tokenizer=tokenizer)
    sampler_config = build_sampler_config(args)
    steps = sampler_config.steps
    num_blocks, effective_steps_per_block = infer_steps_per_block(
        max_new_tokens=args.max_new_tokens,
        steps=steps,
        block_size=args.block_size,
        explicit_steps_per_block=args.steps_per_block,
    )
    if process.is_main_process:
        print(
            "Validated model class: "
            f"{type(model).__module__}.{type(model).__name__}; "
            f"config={type(model.config).__module__}.{type(model.config).__name__}"
        )

    examples, prompts = build_prompt_records(dataset, tokenizer)

    completed_indices = (
        load_completed_dataset_indices(output_path, args.n)
        if args.resume_existing
        else set()
    )
    rank_completed_indices_by_rank = [
        (
            load_completed_dataset_indices(rank_paths[rank], args.n)
            if args.resume_existing and process.world_size > 1
            else set()
        )
        for rank in range(process.world_size)
    ]
    rank_completed_indices = rank_completed_indices_by_rank[process.rank]

    if process.is_main_process and completed_indices:
        print(
            f"Resuming existing output: {len(completed_indices)} completed "
            f"records in {output_path}"
        )
    if rank_completed_indices:
        print(
            f"Rank {process.rank}: resuming {len(rank_completed_indices)} "
            f"rank-local records in {local_output_path}"
        )

    pending_pairs = [
        (example, prompt)
        for example, prompt in zip(examples, prompts)
        if example["_dataset_index"] not in completed_indices
    ]
    pending_pairs_by_rank = [
        [
            (example, prompt)
            for example, prompt in pending_pairs[rank :: process.world_size]
            if example["_dataset_index"] not in rank_completed_indices_by_rank[rank]
        ]
        for rank in range(process.world_size)
    ]
    local_pending_pairs = pending_pairs_by_rank[process.rank]
    if rank_completed_indices:
        local_pending_pairs = [
            (example, prompt)
            for example, prompt in local_pending_pairs
            if example["_dataset_index"] not in rank_completed_indices
        ]
    target_rank_counts = [
        len(rank_completed_indices_by_rank[rank]) + len(pending_pairs_by_rank[rank])
        for rank in range(process.world_size)
    ]
    total_pending_prompts = sum(len(rank_pending) for rank_pending in pending_pairs_by_rank)

    record_count = 0
    file_mode = "a" if args.resume_existing else "w"

    if process.is_main_process:
        print(
            f"Generating {total_pending_prompts} pending prompts x {args.n} samples "
            f"= {total_pending_prompts * args.n} new generations"
        )
        print(
            "BD3LM sampler: "
            f"max_tokens={args.max_new_tokens}, steps={steps}, "
            f"block_size={args.block_size}, num_blocks={num_blocks}, "
            f"steps_per_block={effective_steps_per_block}, "
            f"sample_batch_size={args.sample_batch_size}"
        )
        print(
            f"Launcher processes={process.world_size}; "
            f"per-rank output={'enabled' if process.world_size > 1 else 'disabled'}"
        )
    print(
        f"Rank {process.rank}/{process.world_size}: "
        f"{len(local_pending_pairs)} pending prompts"
    )

    with local_output_path.open(file_mode, encoding="utf-8") as file:
        for example, prompt_ids in tqdm(
            local_pending_pairs,
            desc=f"MiniF2F rank {process.rank}",
            disable=not process.is_main_process,
        ):
            completions, generation_metrics = generate_completions(
                sampler=sampler,
                tokenizer=tokenizer,
                sampler_config=sampler_config,
                prompt_ids=prompt_ids,
                num_samples=args.n,
                sample_batch_size=args.sample_batch_size,
                empty_cache=args.empty_cache,
                show_progress=process.is_main_process,
            )
            record = make_record(example, completions, generation_metrics)
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
            file.flush()
            record_count += 1

    if process.world_size > 1:
        if process.is_main_process:
            wait_for_rank_record_counts(
                rank_paths=rank_paths,
                target_counts=target_rank_counts,
                expected_n=args.n,
            )
            merged_count = merge_rank_outputs(
                output_path=output_path,
                rank_paths=rank_paths,
                expected_n=args.n,
                resume_existing=args.resume_existing,
            )
        else:
            merged_count = 0
    else:
        merged_count = record_count + len(completed_indices)

    metrics_summary = {}
    if process.is_main_process:
        metrics_summary = summarize_generation_metrics(
            output_path=output_path,
            expected_n=args.n,
            cfg_scale=args.cfg_scale,
        )

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "loaded_hf_split": loaded_split,
        "num_examples": merged_count,
        "num_new_examples": max(0, merged_count - len(completed_indices)),
        "num_resumed_examples": len(completed_indices),
        "num_samples_per_example": args.n,
        "total_generations": merged_count * args.n,
        "new_generations": max(0, merged_count - len(completed_indices)) * args.n,
        "model": args.model_name_or_path,
        "max_tokens": args.max_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "steps": steps,
        "steps_per_block": args.steps_per_block,
        "effective_steps_per_block": effective_steps_per_block,
        "num_blocks": num_blocks,
        "block_size": args.block_size,
        "temperature": args.temperature,
        "remasking": args.remasking,
        "stochastic_transfer": args.stochastic_transfer,
        "cfg_scale": args.cfg_scale,
        "right_shift_logits": args.right_shift_logits,
        "sample_batch_size": args.sample_batch_size,
        "half": args.half,
        "quarter": args.quarter,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "split": args.num_shards,
        "quad": args.shard_index,
        "resume_existing": args.resume_existing,
        "seed": args.seed,
        "accelerate_num_processes": process.world_size,
        "rank_output_paths": [str(path) for path in rank_paths] if process.world_size > 1 else [],
        "output_path": str(output_path),
        "generation_metrics": metrics_summary,
        "generation_metrics_notes": {
            "output_tokens": "Tokens decoded after trimming at the first EOS/EOT after the prompt; best for user-visible AR-vs-diffusion throughput.",
            "canvas_tokens": "Tokens appended to the diffusion canvas before EOS trimming; useful for diffusion compute accounting.",
            "observed_diffusion_steps": "Observed denoising model forwards, divided by the CFG multiplier when CFG is enabled.",
            "estimated_flops": "Approximate decoder-only forward FLOPs from config dimensions; includes a separate LM-head term and ignores non-matmul overhead.",
        },
    }
    if process.is_main_process:
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=True)
            file.write("\n")

        print(f"\n[OK] Wrote {merged_count} records to: {output_path}")
        print(f"[OK] Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
