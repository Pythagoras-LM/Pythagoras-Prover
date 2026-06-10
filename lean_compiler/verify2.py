import json
import os
import sys
import glob
import strictfire


import os
import time
import json
import ctypes
import resource
import tempfile
import traceback
import subprocess
import multiprocessing as mp
from tqdm import tqdm

HOME_DIR = os.path.expanduser("~")
DEFAULT_LAKE_PATH = f"{HOME_DIR}/.elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = "/dev/shm/mathlib4"


def lean4_parser(code, ast_data):
    """
    Simplified implementation of the AST parser.
    In the self-contained version, we return the original AST data.
    """
    # For a minimal implementation, we're just passing through the AST data
    return ast_data


def verify_lean4_file(
    code,
    lake_path=DEFAULT_LAKE_PATH,
    lean_workspace=DEFAULT_LEAN_WORKSPACE,
    last_env=None,
    verbose=False,
    timeout=300,
    allTactics=False,
    ast=False,
    premises=False,
    tactics=False,
):
    """Verify a Lean4 file and return the results."""
    command = dict(
        cmd=code,
        allTactics=allTactics,
        ast=ast,
        tactics=tactics,
        premises=premises,
    )
    if last_env is not None:
        command.update(env=last_env)
    message_str = json.dumps(command, ensure_ascii=False)
    if verbose:
        print(message_str)
    start_time = time.time()
    system_messages = ""
    outputs = None
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)

            command = [lake_path, "exe", "repl"]
            outputs = subprocess.run(
                [lake_path, "exe", "repl"],
                stdin=temp_file,
                capture_output=True,
                text=True,
                cwd=lean_workspace,
                timeout=timeout,
            )
        result = json.loads(outputs.stdout)
        ast_results = (
            lean4_parser(code, result["ast"])
            if "ast" in result and result["ast"]
            else {}
        )
        result = {
            "sorries": result.get("sorries", []),
            "tactics": result.get("tactics", []),
            "errors": [
                m
                for m in result.get("messages", [])
                if m["severity"] == "error"
            ],
            "warnings": [
                m
                for m in result.get("messages", [])
                if m["severity"] == "warning"
            ],
            "infos": [
                m
                for m in result.get("messages", [])
                if m["severity"] == "info"
            ],
            "system_messages": system_messages,
            "system_errors": None,
            "ast": ast_results,
            "verified_code": code,
        }
        result["pass"] = not result["errors"]
        result["complete"] = (
            result["pass"]
            and not result["sorries"]
            and not any(
                "declaration uses 'sorry'" in warning["data"]
                or "failed" in warning["data"]
                for warning in result["warnings"]
            )
        )
    except Exception as e:
        error_output = outputs.stdout if outputs is not None else str(e)
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages,
            # "error": outputs.stdout,
            "error": error_output,
        }
    result["verify_time"] = time.time() - start_time
    return result


class ProcessScheduler:
    """
    Base scheduler class that manages a pool of worker processes.
    This is a self-contained implementation based on its usage in Lean4ServerScheduler.
    """

    def __init__(
        self,
        batch_size=512,
        name="scheduler",
        pass_only: bool = False,
        use_tqdm: bool = True,
    ):
        self.name = name
        self.batch_size = batch_size
        self.task_queue = mp.Queue()
        self.request_statuses = mp.Manager().dict()
        self.lock = mp.Lock()
        self.request_counter = mp.Value(ctypes.c_int, 0)
        self.pass_only = pass_only
        self.use_tqdm = use_tqdm

    def _get_request_id(self):
        """Generate a unique request ID."""
        with self.request_counter.get_lock():
            self.request_counter.value += 1
            return f"{self.name}-{self.request_counter.value}"

    def submit_request(self, task):
        """Submit a single request to the task queue."""
        request_id = self._get_request_id()
        self.task_queue.put([(0, request_id, task)])
        return request_id

    def submit_all_request(self, tasks):
        """Submit multiple requests to the task queue."""
        request_ids = []
        for task in tasks:
            request_id = self.submit_request(task)
            request_ids.append(request_id)
        return request_ids

    def get_request_output(self, request_id, timeout=None):
        """Get the output of a specific request."""
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            if request_id in self.request_statuses:
                return self.request_statuses[request_id]
            time.sleep(0.1)
        return None

    def get_all_request_outputs(self, request_ids, timeout=None):
        """Get the outputs of multiple requests."""
        outputs = []
        n_timeout = 0
        n_json_error = 0
        n_error = 0
        n_verified = 0
        total = len(request_ids)

        if self.use_tqdm:
            with tqdm(total=total, desc="Verifying proofs") as pbar:
                for request_id in request_ids:
                    output = self.get_request_output(request_id, timeout)
                    if output["system_errors"] is not None:
                        if "TimeoutExpired" in output["system_errors"]:
                            n_timeout += 1
                        elif (
                            "json.decoder.JSONDecodeError"
                            in output["system_errors"]
                        ):
                            n_json_error += 1
                        else:
                            n_error += 1
                    if self.pass_only and output["pass"]:
                        n_verified += 1
                    elif output["pass"] and output["complete"]:
                        n_verified += 1
                    outputs.append(output)

                    # Update the description with counts and ratios
                    timeout_ratio = n_timeout / (pbar.n + 1) * 100
                    verified_ratio = n_verified / (pbar.n + 1) * 100
                    json_error_ratio = n_json_error / (pbar.n + 1) * 100
                    other_error_ratio = n_error / (pbar.n + 1) * 100

                    pbar.set_description(
                        f"Verifying proofs: {n_verified}/{pbar.n + 1} verified"
                        f" ({verified_ratio:.1f}%),"
                        f" {n_timeout}/{pbar.n + 1} timeout"
                        f" ({timeout_ratio:.1f}%),"
                        f" {n_json_error}/{pbar.n + 1} JSON err"
                        f" ({json_error_ratio:.1f}%),"
                        f" {n_error}/{pbar.n + 1} other err"
                        f" ({other_error_ratio:.1f}%)"
                    )

                    pbar.update(1)
        else:
            for request_id in request_ids:
                output = self.get_request_output(request_id, timeout)
                if output["system_errors"] is not None:
                    if "TimeoutExpired" in output["system_errors"]:
                        n_timeout += 1
                    elif (
                        "json.decoder.JSONDecodeError"
                        in output["system_errors"]
                    ):
                        n_json_error += 1
                    else:
                        n_error += 1
                if self.pass_only and output["pass"]:
                    n_verified += 1
                elif output["pass"] and output["complete"]:
                    n_verified += 1
                outputs.append(output)

            if total > 0:
                timeout_ratio = n_timeout / total * 100
                verified_ratio = n_verified / total * 100
                json_error_ratio = n_json_error / total * 100
                other_error_ratio = n_error / total * 100

                print(
                    f"Verifying proofs: {n_verified}/{total} verified"
                    f" ({verified_ratio:.1f}%),"
                    f" {n_timeout}/{total} timeout"
                    f" ({timeout_ratio:.1f}%),"
                    f" {n_json_error}/{total} JSON err"
                    f" ({json_error_ratio:.1f}%), {n_error}/{total} other"
                    f" err ({other_error_ratio:.1f}%)"
                )
        return outputs

    def close(self):
        """Clean up resources and terminate worker processes."""
        # Send termination signal to all worker processes
        for _ in range(len(getattr(self, "processes", []))):
            self.task_queue.put(None)


class Lean4ServerProcess(mp.Process):
    """Worker process for verifying Lean4 code."""

    def __init__(self, idx, task_queue, request_statuses, lock, extra_args={}):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock

        self.timeout = extra_args.get("timeout", 300)
        self.memory_limit = extra_args.get("memory_limit", -1)
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)

    def run(self):
        """Main worker process loop."""
        if self.memory_limit > 0:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit * (1000**3), self.memory_limit * (1000**3)),
            )
        while True:
            inputs = self.task_queue.get()
            if inputs is None:  # Terminate when receiving None
                break
            for _, request_id, task in inputs:
                if isinstance(task, str):
                    task = dict(code=task)
                if "timeout" not in task:
                    task["timeout"] = self.timeout
                result = verify_lean4_file(**task)
                if len(result["system_messages"]) > 0:
                    retry_start_time = time.time()
                    while (
                        "lean::exception: failed to create thread"
                        in result["system_messages"]
                        or "std::bad_alloc: std::bad_alloc"
                        in result["system_messages"]
                        or "Cannot allocate memory"
                        in result["system_messages"]
                    ) and time.time() - retry_start_time < self.timeout:
                        time.sleep(0.1)
                        result = verify_lean4_file(**task)
                with self.lock:
                    self.request_statuses[request_id] = result
                    self.last_output_time.value = time.time()
                    self.complete_count.value += 1


class Lean4ServerScheduler(ProcessScheduler):
    """Scheduler for managing Lean4 verification worker processes."""

    def __init__(
        self,
        max_concurrent_requests=64,
        timeout=300,
        memory_limit=-1,
        name="verifier",
        pass_only=False,
        use_tqdm: bool = True,
    ):
        super().__init__(
            batch_size=1, name=name, pass_only=pass_only, use_tqdm=use_tqdm
        )

        self.processes = [
            Lean4ServerProcess(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                extra_args={
                    "timeout": timeout,
                    "memory_limit": memory_limit,
                },
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        print(f"Complete launching {len(self.processes)} LeanServerProcesses")

        self.timeout = timeout
        self._running_monitor = mp.Value(ctypes.c_bool, True)
        self._last_complete_count = mp.Value(ctypes.c_int, 0)
        self._monitor_process = mp.Process(target=self._monitor)
        self._monitor_process.start()

    def _monitor(self):
        """Monitor process to kill hung processes."""
        while self._running_monitor.value:
            time.sleep(1.0)
            subprocess.run(
                ["killall", "repl", f"--older-than={int(self.timeout) + 10}s"],
                capture_output=True,
            )

    def close(self):
        """Clean up resources and terminate all processes."""
        super().close()
        for p in self.processes:
            p.join()
        self._running_monitor.value = False
        self._monitor_process.join()
        print(f"All {len(self.processes)} LeanServerProcesses stopped")


def find_latest_inference_file(
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    n_samples: int = 4,
    use_json_format: bool = True,
    prompt_version: int = 1,
    formalize_question_only: bool = False,
):
    """Find the latest inference file for the given model parameters.

    Args:
        model_name: Name of the model used for inference
        temperature: Temperature used for sampling
        max_tokens: Maximum tokens used
        n_samples: Number of samples per prompt used
        use_json_format: Whether to use JSON format
        prompt_version: Version of the prompt
        formalize_question_only: Whether the outputs contain question formalizations only (True) or full formalizations (False)

    Returns:
        The path to the latest matching inference file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    model_last_name = model_name.split("/")[-1]

    # Look for files matching the pattern
    json_format_str = "json" if use_json_format else "raw"
    formalize_type = "question-only" if formalize_question_only else "full"
    pattern = f"{output_dir}/Autoformalise-fs_{model_last_name}_temp-{temperature}_tokens-{max_tokens}_samples-{n_samples}_format-{json_format_str}_prompt-v{prompt_version}_formalize-{formalize_type}_*.jsonl"
    matching_files = glob.glob(pattern)

    # Filter out files containing "part" in their filename
    matching_files = [f for f in matching_files if ".part" not in f]

    if not matching_files:
        raise FileNotFoundError(
            f"No inference files found matching pattern: {pattern}"
        )

    # Sort by modification time (newest first)
    latest_file = max(matching_files, key=os.path.getmtime)
    return latest_file


def main(
    model_name: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    n_samples: int = 4,
    input_file: str = None,
    add_instruction: bool = False,
    use_json_format: bool = True,
    prompt_version: int = 1,
    formalize_question_only: bool = False,
    n_cpus: int = 4,
):
    """Verify generated proofs/formalizations and update the jsonl file.

    Args:
        model_name: Name of the model used for inference
        temperature: Temperature used for sampling
        max_tokens: Maximum tokens used
        n_samples: Number of samples per prompt
        input_file: Path to the jsonl file (optional, will be deduced if not provided)
        add_instruction: Whether the outputs were generated with add_instruction=True
        use_json_format: Whether the outputs were generated with use_json_format=True
        prompt_version: Version of the prompt used
        formalize_question_only: Whether the outputs contain question formalizations only (True) or full formalizations (False)
        n_cpus: Number of CPUs to use for verification
    """
    # If input file is not provided, find the latest file for the model
    if input_file is None:
        input_file = find_latest_inference_file(
            model_name,
            temperature,
            max_tokens,
            n_samples,
            use_json_format,
            prompt_version,
            formalize_question_only,
        )

    print(f"Loading generated responses from {input_file}")

    # Load the jsonl file
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    # Collect all parsed formal content for verification
    all_formal_content = []
    content_indices = (
        []
    )  # To track which prompt and sample each content belongs to

    n_removed = 0
    total_content = 0
    for prompt_idx, entry in enumerate(data):
        # Check if the entry has use_json_format field, otherwise use the provided parameter
        entry_use_json_format = entry.get("use_json_format", use_json_format)

        for sample_idx, content in enumerate(entry["parsed_proofs"]):
            # Only add to verification queue if we could parse the content
            if content is not None:
                total_content += 1
                # some completions might start with `Formal proof:` or `Formalisation:`
                # although parse_model_output should handle this, double check
                if content.startswith("Formal proof:"):
                    content = content[len("Formal proof:") :].strip()
                    n_removed += 1
                elif content.startswith("Formalisation:"):
                    content = content[len("Formalisation:") :].strip()
                    n_removed += 1

                all_formal_content.append(content)
                content_indices.append((prompt_idx, sample_idx))

    # number and ratio of removed prefixes
    ratio = n_removed / total_content if total_content > 0 else 0
    print(
        "Removed prefixes ('Formal proof:', 'Formalisation:') in"
        f" {n_removed}/{total_content} parsed contents ({ratio:.2%})"
    )

    # Verify all formal content in parallel
    print(
        f"Verifying {len(all_formal_content)} generated formal content with"
        " Lean4..."
    )
    lean4_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=n_cpus,
        timeout=600,
        memory_limit=10,
        name="verifier",
    )
    request_id_list = lean4_scheduler.submit_all_request(all_formal_content)
    verification_results = lean4_scheduler.get_all_request_outputs(
        request_id_list
    )

    # Map verification results back to their corresponding completions
    for (prompt_idx, sample_idx), result in zip(
        content_indices, verification_results
    ):
        # Update the verification result in our data
        data[prompt_idx]["verification_results"][sample_idx] = result

    # Write the updated data back to the file
    print(f"Updating {input_file} with verification results")
    with open(input_file, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    # Calculate and print verification statistics
    verified_count = 0
    total_count = 0

    for entry in data:
        for result in entry["verification_results"]:
            if result is not None:
                total_count += 1
                # 'complete' means no errors or sorries, which is a good check
                # for both proofs and formal statements
                if result.get("complete", False):
                    verified_count += 1

    if total_count > 0:
        print(
            f"Verification complete: {verified_count}/{total_count} formal"
            f" contents verified ({verified_count/total_count:.2%})"
        )
    else:
        print("No formal content was verified")
    lean4_scheduler.close()


if __name__ == "__main__":
    strictfire.StrictFire(main)
