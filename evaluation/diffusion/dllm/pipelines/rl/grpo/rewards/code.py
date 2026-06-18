"""Coding task reward functions."""

import multiprocessing
import os
import random
import re
import resource
import string
import time


def run_test(test_func_name, code_str, result_dict, cwd_path, rank):
    cwd_path = cwd_path + "/" + str(rank)
    os.makedirs(cwd_path, exist_ok=True)

    def target():
        try:
            soft, hard = 1_000_000_000, 1_000_000_000
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
            os.chdir(cwd_path)
            local_ns = {}
            exec(code_str, local_ns)
            local_ns[test_func_name]()
            result_dict[test_func_name] = True
        except Exception:
            result_dict[test_func_name] = False

    proc = multiprocessing.Process(target=target)
    proc.start()
    proc.join(timeout=1.0)

    if proc.is_alive():
        proc.terminate()
        result_dict[test_func_name] = False


def split_test_function(test_code: str, base_name: str = "test_case"):
    lines = test_code.strip().splitlines()
    result = []
    counter = 1

    for line in lines:
        line = line.strip()
        if line.startswith("assert"):
            fn = f"def {base_name}_{counter}():\n    {line}"
            result.append(fn)
            counter += 1

    return "\n\n".join(result)


BLOCKED_MODULES = [
    " os",
    " sys",
    " subprocess",
    " shutil",
    " socket",
    " psutil",
    " ctypes",
    " pathlib",
    " builtins",
    "__import__",
]


def is_safe_code(code_str):
    for blocked in BLOCKED_MODULES:
        if blocked in code_str:
            return False
    return True


def time_based_random_string(length=10):
    seed = int(time.time() * 1e6)
    random.seed(seed)
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def coding_reward_func(
    prompts, completions, answer, step=None, run_name=None, verbose=False, **kwargs
) -> list[float]:
    execution_cwd = kwargs.get("cwd_path") or os.path.join(
        os.getcwd(), "temp_coding_reward"
    )
    if not os.path.exists(execution_cwd):
        os.makedirs(execution_cwd, exist_ok=True)
    programs = []
    for group in completions:
        for message in group:
            content = message["content"]

            code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            code = code_match.group(1) if code_match else ""

            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            is_in_answer = False
            if answer_match and code:
                answer_content = answer_match.group(1)
                is_in_answer = f"```python\n{code}```" in answer_content

            programs.append((code, is_in_answer))

    unit_tests = [entry["tests"] for entry in answer]
    rewards = []
    for i, (solution, tests) in enumerate(zip(programs, unit_tests)):
        solution, is_in_answer = solution
        is_in_answer_reward = 0.5 if is_in_answer else 0
        import_match = re.search(r"from solution import (\w+)", tests)
        if not import_match:
            assert_match = re.search(r"assert\s+(\w+)\s*\(", tests)
            if assert_match:
                imported_func = assert_match.group(1)
            else:
                if verbose:
                    print("No import or assert-based function name found in test code.")
                    print("=" * 10, "tests", "=" * 10)
                    print(tests)
                    print("=" * 30)
                rewards.append(0)
                continue
        else:
            imported_func = import_match.group(1)

        solution_match = re.search(r"def (\w+)\(", solution)
        if not solution_match:
            if verbose:
                print("No function definition in generation")
                print("=" * 10, "model output", "=" * 10)
                print(solution)
                print("=" * 10)
            rewards.append(0 + is_in_answer_reward)
            continue
        defined_func = solution_match.group(1)

        if defined_func != imported_func:
            solution = re.sub(
                rf"\bdef {defined_func}\b", f"def {imported_func}", solution
            )

        if not is_safe_code(solution):
            rewards.append(0 + is_in_answer_reward)
            if verbose:
                print(f"Potentially Unsafe Generation:\n{solution}")
            continue

        test_funcs = re.findall(r"def (\w+)\(\):", tests)

        if len(test_funcs) <= 1:
            tests = split_test_function(tests)
            if verbose:
                print(f"Fixed Test functions\n{tests}")
            test_funcs = re.findall(r"def (\w+)\(\):", tests)

        if import_match:
            test_code = re.sub(r"from solution import \w+", lambda _: solution, tests)
        elif assert_match:
            test_code = solution + tests

        manager = multiprocessing.Manager()
        result_dict = manager.dict()

        jobs = []
        for rank, fn in enumerate(test_funcs):
            p = multiprocessing.Process(
                target=run_test, args=(fn, test_code, result_dict, execution_cwd, rank)
            )
            p.start()
            jobs.append(p)

        for p in jobs:
            p.join()

        passed = sum(result_dict.get(fn, False) for fn in test_funcs)
        total = len(test_funcs)
        reward = passed / total if total > 0 else 0.0
        rewards.append(reward + is_in_answer_reward)
    return rewards
