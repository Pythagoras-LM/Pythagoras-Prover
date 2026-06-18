"""Countdown task reward functions."""

import re


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except Exception:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def compute_score(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.1,
    score=1.0,
    verbose=False,
):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)

    if verbose:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if verbose:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if verbose:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if verbose:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if verbose:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if verbose:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except Exception:
        if verbose:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(
    prompts, completions, run_name=None, step=None, rank=None, verbose=False, **kwargs
) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth, verbose=verbose))

    return scores
