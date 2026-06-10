"""Verify generated Lean proofs against their target formal statements."""

import json
import re
from statement_match import STATEMENT_MATCH_EXACT, STATEMENT_MATCH_MODES, make_statement_mismatch_result, prepare_proof_for_statement_match
from verify2 import Lean4ServerScheduler

PROOF_SECTION_MARKER = "### Complete Lean 4 Proof"
LEAN_BLOCK_RE = re.compile(r"```lean(?:4)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
OPEN_LEAN_FENCE_RE = re.compile(r"```lean(?:4)?\s*\n", re.IGNORECASE)
DEFAULT_MINIF2F_HEADER = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
""".strip()
DEFAULT_INPUT_FILE = None
DEFAULT_TIMEOUT = 120


def load_input_data(input_file: str):
    if input_file.endswith(".jsonl"):
        with open(input_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def default_output_path(input_file: str) -> str:
    if input_file.endswith(".jsonl"):
        return input_file[:-6] + "_verified.json"
    if input_file.endswith(".json"):
        return input_file[:-5] + "_verified.json"
    return input_file + "_verified.json"


def extract_complete_lean_proof_section(text: str | None) -> str | None:
    if not text:
        return None
    marker_index = text.rfind(PROOF_SECTION_MARKER)
    if marker_index == -1:
        return None
    section = text[marker_index:].strip()
    return section or None


def extract_last_lean_block(text: str | None) -> str | None:
    if not text:
        return None
    marker_index = text.rfind(PROOF_SECTION_MARKER)
    search_text = text[marker_index:] if marker_index != -1 else text
    matches = LEAN_BLOCK_RE.findall(search_text)
    if matches:
        return matches[-1].strip() or None

    # Fall back to an unclosed trailing Lean fence when generation ends before
    # the closing ``` token is emitted.
    open_fences = list(OPEN_LEAN_FENCE_RE.finditer(search_text))
    if not open_fences:
        return None

    trailing_block = search_text[open_fences[-1].end() :].strip()
    return trailing_block or None


def resolve_verification_header(header: str | None) -> str:
    stripped = (header or "").strip()
    if stripped.startswith("import "):
        return stripped
    return DEFAULT_MINIF2F_HEADER


def looks_like_lean_source(text: str | None) -> bool:
    if not text:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    disallowed_markers = ("```", "### ", "<|im_", "<think>", "</think>", "**Problem", "**Approach")
    if any(marker in stripped for marker in disallowed_markers):
        return False

    if stripped.startswith("import "):
        return True

    lean_markers = ("theorem ", "lemma ", ":= by", "\nby\n", "\n  by\n")
    return any(marker in stripped for marker in lean_markers)


def proof_and_result_field_names(entry: dict, index: int) -> tuple[str, str]:
    if "merged_proof_1" in entry:
        return f"merged_proof_{index}", f"proof_verification_result_{index}"
    return "merged_proof", "proof_verification_result"


def build_verification_source(entry: dict, sample: dict) -> str | None:
    source = sample.get("verification_source")
    if source is None:
        source = sample.get("lean_code_block")
    if source is None:
        source = extract_last_lean_block(sample.get("completion"))
    if source is None:
        source = extract_last_lean_block(sample.get("raw_completion"))
    if source is None:
        candidate_source = sample.get("candidate_source")
        if looks_like_lean_source(candidate_source):
            source = candidate_source
    if source is None:
        return None

    source = source.strip()
    if source.startswith("import "):
        return source

    if "theorem " in source or "lemma " in source:
        proof_text = source
    else:
        proof_text = f"{(entry.get('formal_statement') or '').rstrip()}\n{source}".strip()

    return f"{resolve_verification_header(entry.get('header'))}\n\n{proof_text}".strip()


def normalize_entries(data: list[dict]) -> list[dict]:
    normalized = []
    for entry in data:
        if "samples" not in entry:
            normalized.append(entry)
            continue

        normalized_entry = {k: v for k, v in entry.items() if k != "samples"}
        samples = entry.get("samples", [])
        normalized_entry["n"] = len(samples)

        for i, sample in enumerate(samples, start=1):
            normalized_entry[f"merged_proof_{i}"] = build_verification_source(entry, sample)
            normalized_entry[f"complete_lean_proof_section_{i}"] = sample.get("complete_lean_proof_section") or extract_complete_lean_proof_section(sample.get("completion"))
            normalized_entry[f"reasoning_{i}"] = sample.get("reasoning")
            normalized_entry[f"completion_{i}"] = sample.get("completion")
            normalized_entry[f"raw_completion_{i}"] = sample.get("raw_completion")

        normalized.append(normalized_entry)

    return normalized


def verify_file(input_file: str, output_file: str | None = None, n_cpus: int = 16, timeout: int = DEFAULT_TIMEOUT, statement_match: str = STATEMENT_MATCH_EXACT):
    """
    Verify Lean 4 proofs (complete proofs without sorry).
    
    Args:
        input_file: Path to the input JSON/JSONL file
        output_file: Path to save the output (defaults to input_file with _verified suffix)
        n_cpus: Number of CPUs to use for parallel verification
        timeout: Timeout in seconds for each verification
        statement_match: "exact" fails mismatched statements before Lean verification;
            "substitute" replaces a mismatched theorem declaration with the
            target formal statement before verification.
    """
    if statement_match not in STATEMENT_MATCH_MODES:
        raise ValueError(f"--statement_match must be one of {STATEMENT_MATCH_MODES}, got {statement_match!r}")
    # Default output file name
    if output_file is None:
        output_file = default_output_path(input_file)
    
    print(f"Loading data from {input_file}")
    data = load_input_data(input_file)
    data = normalize_entries(data)
    
    print(f"Found {len(data)} entries to verify")
    
    # Extract all merged_proof fields and track which need verification.
    proofs_to_verify = []
    entries_to_verify = []
    proof_field_names = []
    null_count = 0
    statement_mismatch_count = 0
    statement_substitution_count = 0

    print(f"Statement match mode: {statement_match}")

    for entry in data:
        has_formal_statement = bool((entry.get('formal_statement') or '').strip())

        if 'merged_proof' in entry:
            proof = entry.get('merged_proof')
            result_field = 'proof_verification_result'
            if proof is None or proof == "":
                null_count += 1
                entry[result_field] = {"pass": False, "complete": False, "errors": [{"data": "merged_proof is null or empty"}], "warnings": [], "sorries": [], "verify_time": 0, "system_errors": None}
                continue

            proof, original_match, final_match, substituted = prepare_proof_for_statement_match(entry, proof, statement_match)
            entry['merged_proof'] = proof
            entry['formal_statement_match_original'] = original_match
            entry['formal_statement_match'] = final_match
            entry['formal_statement_substituted'] = substituted
            statement_substitution_count += int(substituted)

            if has_formal_statement and not final_match:
                statement_mismatch_count += 1
                entry[result_field] = make_statement_mismatch_result(statement_match, original_match, final_match, substituted)
                continue

            proofs_to_verify.append(proof)
            entries_to_verify.append(entry)
            proof_field_names.append('merged_proof')

        elif 'merged_proof_1' in entry:
            n = entry.get('n', 1)
            for i in range(1, n + 1):
                field_name = f'merged_proof_{i}'
                result_field = f'proof_verification_result_{i}'
                proof = entry.get(field_name)

                if proof is None or proof == "":
                    null_count += 1
                    entry[result_field] = {"pass": False, "complete": False, "errors": [{"data": f"{field_name} is null or empty"}], "warnings": [], "sorries": [], "verify_time": 0, "system_errors": None}
                    continue

                proof, original_match, final_match, substituted = prepare_proof_for_statement_match(entry, proof, statement_match)
                entry[field_name] = proof
                entry[f'formal_statement_match_original_{i}'] = original_match
                entry[f'formal_statement_match_{i}'] = final_match
                entry[f'formal_statement_substituted_{i}'] = substituted
                statement_substitution_count += int(substituted)

                if has_formal_statement and not final_match:
                    statement_mismatch_count += 1
                    entry[result_field] = make_statement_mismatch_result(statement_match, original_match, final_match, substituted)
                    continue

                proofs_to_verify.append(proof)
                entries_to_verify.append(entry)
                proof_field_names.append(field_name)
        else:
            null_count += 1
            entry['proof_verification_result'] = {"pass": False, "complete": False, "errors": [{"data": "merged_proof is null or empty"}], "warnings": [], "sorries": [], "verify_time": 0, "system_errors": None}

    print(f"Found {null_count} null/missing proof slot(s) (marked as failed)")
    print(f"Statement substitutions applied: {statement_substitution_count}")
    print(f"Statement mismatches marked failed: {statement_mismatch_count}")
    
    # Verify only valid proofs
    if proofs_to_verify:
        print(f"Verifying {len(proofs_to_verify)} Lean 4 proofs (complete proofs required, no sorry allowed)...")
        print(f"Settings: n_cpus={n_cpus}, timeout={timeout}s")
        
        lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=n_cpus, timeout=timeout, memory_limit=10, name="verifier", pass_only=False)
        
        request_ids = lean4_scheduler.submit_all_request(proofs_to_verify)
        verification_results = lean4_scheduler.get_all_request_outputs(request_ids)
        
        # Add verification results to data for verified entries
        for entry, result, field_name in zip(entries_to_verify, verification_results, proof_field_names):
            # Save result with appropriate field name
            # For merged_proof -> proof_verification_result
            # For merged_proof_1 -> proof_verification_result_1, etc.
            if field_name == 'merged_proof':
                result_field = 'proof_verification_result'
            else:
                # Extract the number from merged_proof_N
                num = field_name.split('_')[-1]
                result_field = f'proof_verification_result_{num}'
            entry[result_field] = {"pass": result.get("pass", False), "complete": result.get("complete", False), "errors": result.get("errors", []), "warnings": result.get("warnings", []), "sorries": result.get("sorries", []), "verify_time": result.get("verify_time", 0), "system_errors": result.get("system_errors", None)}
        
        lean4_scheduler.close()
    else:
        print("No valid proofs to verify")
    
    # Save results
    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    # Calculate detailed statistics per proof
    print(f"\n{'='*80}")
    print("Detailed Proof Verification Results:")
    print(f"{'='*80}")
    
    # Count per-proof statistics
    max_n = max((entry.get('n', 1) for entry in data), default=1)
    total_expected = sum(entry.get('n', 1) for entry in data)
    proof_stats = {}
    
    for i in range(1, max_n + 1):
        proof_stats[i] = {'total': 0, 'verified': 0, 'complete': 0, 'pass': 0, 'error': 0, 'null': 0}
    
    for entry in data:
        entry_n = entry.get('n', 1)
        for i in range(1, entry_n + 1):
            merged_field, result_field = proof_and_result_field_names(entry, i)
            
            # Check if proof exists
            if merged_field not in entry or entry[merged_field] is None or entry[merged_field] == "":
                proof_stats[i]['null'] += 1
                proof_stats[i]['total'] += 1
                continue
            
            proof_stats[i]['total'] += 1
            
            if result_field in entry:
                proof_stats[i]['verified'] += 1
                result = entry[result_field]
                if result.get('complete', False):
                    proof_stats[i]['complete'] += 1
                if result.get('pass', False):
                    proof_stats[i]['pass'] += 1
                else:
                    proof_stats[i]['error'] += 1
    
    # Print per-proof statistics
    total_problems = len(data)
    print(f"\nPer-Proof Statistics ({total_problems} problems):")
    print(f"{'-'*80}")
    print(f"{'Proof':<8} {'Valid':<8} {'Verified':<10} {'Complete':<10} {'Pass':<8} {'Error':<8} {'Null':<8}")
    print(f"{'-'*80}")
    
    total_valid = 0
    total_verified = 0
    total_complete = 0
    total_pass = 0
    total_error = 0
    total_null = 0
    
    for i in range(1, max_n + 1):
        stats = proof_stats[i]
        valid = stats['total'] - stats['null']
        total_valid += valid
        total_verified += stats['verified']
        total_complete += stats['complete']
        total_pass += stats['pass']
        total_error += stats['error']
        total_null += stats['null']
        
        print(f"Proof {i:<3} {valid:<8} {stats['verified']:<10} {stats['complete']:<10} {stats['pass']:<8} {stats['error']:<8} {stats['null']:<8}")
    
    print(f"{'-'*80}")
    print(f"{'TOTAL':<8} {total_valid:<8} {total_verified:<10} {total_complete:<10} {total_pass:<8} {total_error:<8} {total_null:<8}")
    
    # Calculate pass@k
    problems_with_complete = 0
    problems_with_pass = 0
    
    for entry in data:
        entry_n = entry.get('n', 1)
        has_complete = False
        has_pass = False
        
        for i in range(1, entry_n + 1):
            _, result_field = proof_and_result_field_names(entry, i)
            if result_field in entry:
                if entry[result_field].get('complete', False):
                    has_complete = True
                if entry[result_field].get('pass', False):
                    has_pass = True
        
        if has_complete:
            problems_with_complete += 1
        if has_pass:
            problems_with_pass += 1
    
    print(f"\n{'='*80}")
    print("Complete/Pass Metrics (binary: at least 1 successful proof per problem):")
    print(f"{'='*80}")
    print(f"  Total Problems: {total_problems}")
    print(f"  Complete@{max_n}: {problems_with_complete}/{total_problems} ({problems_with_complete/total_problems*100:.1f}%)")
    print(f"  Pass@{max_n}: {problems_with_pass}/{total_problems} ({problems_with_pass/total_problems*100:.1f}%)")
    print(f"\nOverall Statistics:")
    print(f"  Total Proofs Expected: {total_expected}")
    print(f"  Valid Proofs: {total_valid} ({total_valid/total_expected*100:.1f}%)" if total_expected > 0 else "  Valid Proofs: 0 (0.0%)")
    print(f"  Null/Empty Proofs: {total_null} ({total_null/total_expected*100:.1f}%)" if total_expected > 0 else "  Null/Empty Proofs: 0 (0.0%)")
    print(f"  Verified: {total_verified}/{total_valid} ({total_verified/total_valid*100:.1f}% of valid)" if total_valid > 0 else "  Verified: 0/0 (0.0% of valid)")
    print(f"  Complete: {total_complete}/{total_verified} ({total_complete/total_verified*100:.1f}% of verified)" if total_verified > 0 else "  Complete: 0")
    print(f"  Pass: {total_pass}/{total_verified} ({total_pass/total_verified*100:.1f}% of verified)" if total_verified > 0 else "  Pass: 0")
    print(f"  Error: {total_error}/{total_verified} ({total_error/total_verified*100:.1f}% of verified)" if total_verified > 0 else "  Error: 0")
    print(f"{'='*80}")
    
    print(f"\nDone! Results saved to: {output_file}")
    return output_file


def resolve_file_jobs(input_file: str | None, input_files: list[str] | None, output_file: str | None, output_files: list[str] | None) -> list[tuple[str, str | None]]:
    files: list[str] = []
    if input_files:
        files.extend(input_files)
    elif input_file:
        files.append(input_file)

    if not files:
        raise ValueError("At least one input file must be provided via --input_file or --input_files.")

    if output_file and len(files) > 1:
        raise ValueError("--output_file can only be used with a single input file. Use --output_files for multiple inputs.")

    if output_files is not None and len(output_files) != len(files):
        raise ValueError("--output_files must have the same number of paths as --input_files.")

    if output_files is not None:
        return list(zip(files, output_files))

    if output_file is not None:
        return [(files[0], output_file)]

    return [(file_path, None) for file_path in files]


def main(input_file: str | None = DEFAULT_INPUT_FILE, input_files: list[str] | None = None, output_file: str | None = None, output_files: list[str] | None = None, n_cpus: int = 16, timeout: int = DEFAULT_TIMEOUT, statement_match: str = STATEMENT_MATCH_EXACT):
    jobs = resolve_file_jobs(input_file=input_file, input_files=input_files, output_file=output_file, output_files=output_files)

    print(f"Preparing to verify {len(jobs)} input file(s)")
    for index, (current_input, current_output) in enumerate(jobs, start=1):
        print(f"\n{'#' * 80}")
        print(f"File {index}/{len(jobs)}: {current_input}")
        if current_output is not None:
            print(f"Requested output: {current_output}")
        print(f"{'#' * 80}")
        verify_file(input_file=current_input, output_file=current_output, n_cpus=n_cpus, timeout=timeout, statement_match=statement_match)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Lean 4 proofs (complete proofs without sorry)")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE, help="Path to a single input JSON/JSONL file from evaluation output")
    parser.add_argument("--input_files", nargs="+", default=None, help="Paths to multiple input JSON/JSONL files")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output JSON file for a single input (default: input_file with _verified suffix)")
    parser.add_argument("--output_files", nargs="+", default=None, help="Paths to output JSON files for multiple inputs")
    parser.add_argument("--n_cpus", type=int, default=16, help="Number of CPUs for parallel verification")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds for each verification")
    parser.add_argument("--statement_match", "--statement-match", choices=STATEMENT_MATCH_MODES, default=STATEMENT_MATCH_EXACT, help="How to handle theorem-statement mismatch: 'exact' marks mismatches failed; 'substitute' replaces the generated declaration with formal_statement before verification.")
    
    args = parser.parse_args()
    main(input_file=args.input_file, input_files=args.input_files, output_file=args.output_file, output_files=args.output_files, n_cpus=args.n_cpus, timeout=args.timeout, statement_match=args.statement_match)
