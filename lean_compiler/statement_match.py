"""Utilities for matching and substituting Lean theorem statements."""

import re

LEAN_BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)
LEAN_DECL_RE = re.compile(r"\b(theorem|lemma)\s+([^\s:]+)")
LEAN_PROOF_START_RE = re.compile(r":=\s*by\b")
LEAN_SORRY_RE = re.compile(r"\bsorry\b")
WHITESPACE_RE = re.compile(r"\s+")

STATEMENT_MATCH_EXACT = "exact"
STATEMENT_MATCH_SUBSTITUTE = "substitute"
STATEMENT_MATCH_MODES = (STATEMENT_MATCH_EXACT, STATEMENT_MATCH_SUBSTITUTE)


def canonicalize_lean_for_statement_match(text: str | None, *, statement_only: bool) -> str | None:
    if not text:
        return None

    normalized = LEAN_BLOCK_COMMENT_RE.sub("", text)
    normalized = LEAN_SORRY_RE.sub("", normalized)

    declaration = LEAN_DECL_RE.search(normalized)
    if declaration is not None:
        name_start, name_end = declaration.span(2)
        normalized = f"{normalized[:name_start]}__THEOREM_NAME__{normalized[name_end:]}"

        if statement_only:
            proof_start = LEAN_PROOF_START_RE.search(normalized, declaration.end())
            if proof_start is not None:
                normalized = normalized[: proof_start.end()]

    return WHITESPACE_RE.sub(" ", normalized).strip()


def formal_statement_matches_proof(formal_statement: str | None, proof: str | None) -> bool:
    formal = canonicalize_lean_for_statement_match(formal_statement, statement_only=True)
    proof_source = canonicalize_lean_for_statement_match(proof, statement_only=False)
    return bool(formal and proof_source and formal in proof_source)


def split_statement_and_body(source: str | None) -> tuple[str, str, str] | None:
    if not source:
        return None

    declaration = LEAN_DECL_RE.search(source)
    if declaration is None:
        return None

    proof_start = LEAN_PROOF_START_RE.search(source, declaration.end())
    if proof_start is None:
        return None

    return (
        source[: declaration.start()],
        source[declaration.start() : proof_start.end()],
        source[proof_start.end() :],
    )


def substitute_formal_statement_into_proof(formal_statement: str | None, proof: str | None) -> tuple[str | None, bool]:
    if not proof:
        return proof, False

    target_parts = split_statement_and_body(formal_statement)
    proof_parts = split_statement_and_body(proof)
    if target_parts is None or proof_parts is None:
        return proof, False

    target_statement = target_parts[1].strip()
    proof_prefix, proof_statement, proof_body = proof_parts
    if not target_statement or proof_statement.strip() == target_statement:
        return proof, False

    separator = "\n" if proof_prefix.strip() else ""
    substituted = f"{proof_prefix.rstrip()}{separator}{target_statement}{proof_body}".strip()
    return substituted, substituted != proof


def prepare_proof_for_statement_match(entry: dict, proof: str | None, statement_match: str) -> tuple[str | None, bool, bool, bool]:
    original_match = formal_statement_matches_proof(entry.get("formal_statement"), proof)
    prepared_proof = proof
    substituted = False

    if statement_match == STATEMENT_MATCH_SUBSTITUTE and not original_match:
        prepared_proof, substituted = substitute_formal_statement_into_proof(entry.get("formal_statement"), proof)

    final_match = formal_statement_matches_proof(entry.get("formal_statement"), prepared_proof)
    return prepared_proof, original_match, final_match, substituted


def make_statement_mismatch_result(statement_match: str, original_match: bool, final_match: bool, substituted: bool) -> dict:
    return {
        "pass": False,
        "complete": False,
        "errors": [{"data": f"formal statement does not match proof (statement_match={statement_match})"}],
        "warnings": [],
        "sorries": [],
        "verify_time": 0,
        "system_errors": None,
        "statement_match_failed": True,
        "formal_statement_match_original": original_match,
        "formal_statement_match": final_match,
        "formal_statement_substituted": substituted,
    }
