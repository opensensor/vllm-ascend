"""Offline regression checks for GPQA scoring, request parameters, and resume."""

import importlib.util
import io
import json
import socket
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[3] / "tools/gpqa_strata/experiment.py"
SPEC = importlib.util.spec_from_file_location("strata_experiment", MODULE_PATH)
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)


class ExperimentTests(unittest.TestCase):
    def test_port_check_refuses_an_existing_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaises(OSError):
                EXPERIMENT.check_port_available(listener.getsockname()[1])

    def test_port_check_allows_a_recently_closed_connection(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port)) as client:
                connection, _ = listener.accept()
                connection.close()  # Server closes first, leaving its port in TIME_WAIT.
                self.assertEqual(client.recv(1), b"")
        EXPERIMENT.check_port_available(port)

    def test_server_launch_explicitly_selects_real_engine(self):
        # The upstream frontend defaults to mock even when --config is given.
        # Exercise the actual evaluate launch path so omitting --engine cannot
        # silently turn the hardware benchmark into a mock-server run.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Strata"
            (source / "serve").mkdir(parents=True)
            (source / "serve/server.py").write_text("# source fixture\n")
            executable = source / "strata"
            executable.write_text("engine fixture\n")
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            baseline = root / "baseline.jsonl"
            baseline.write_text("baseline fixture\n")
            args = types.SimpleNamespace(root=root, output=root / "results", baseline=baseline, timeout=5)
            config = {
                "exe": str(executable),
                "tokenizer": str(tokenizer),
                "port": 0,
                "experiment": {"model_files": [], "runtime_files": []},
            }

            def check_output(command, **kwargs):
                if "status" in command:
                    return b""
                if "rev-parse" in command:
                    return "revision\n"
                return "NVIDIA RTX PRO 6000\n"

            with (
                patch.object(EXPERIMENT.subprocess, "check_output", side_effect=check_output),
                patch.object(EXPERIMENT.subprocess, "Popen", side_effect=RuntimeError("captured launch")) as launch,
                self.assertRaisesRegex(RuntimeError, "captured launch"),
            ):
                EXPERIMENT.evaluate([{"id": 0, "gold": "A", "prompt": "test"}], args, config, {})
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--engine") + 1], "strata")

    def test_invalid_responses_are_not_successes(self):
        case = {"id": 0, "gold": "A", "prompt": "test"}
        for finish, tokens in (("error", 1), ("length", 0), ("stop", 8193), ("stop", None)):
            payload = {
                "choices": [{"message": {"content": "Answer: A"}, "finish_reason": finish}],
                "usage": {"completion_tokens": tokens},
            }
            with (
                self.subTest(finish=finish, tokens=tokens),
                patch.object(EXPERIMENT.urllib.request, "urlopen", return_value=io.StringIO(json.dumps(payload))),
                self.assertRaises(ValueError),
            ):
                EXPERIMENT.completion(case, "http://localhost", "test", 5)

    def test_split_reasoning_does_not_count_as_final(self):
        observed = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                observed.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = {
                    "choices": [
                        {"message": {"reasoning_content": "Maybe Answer: c", "content": ""}, "finish_reason": "length"}
                    ],
                    "usage": {"completion_tokens": 8192},
                }
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            case = {"id": 0, "gold": "C", "prompt": "test"}
            row = EXPERIMENT.completion(case, f"http://127.0.0.1:{server.server_port}", "test-model", 5)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertEqual(row["answer"], "C")
        self.assertIsNone(row["final_answer"])
        self.assertEqual(observed[0]["max_tokens"], 8192)
        self.assertEqual(observed[0]["temperature"], 0)
        self.assertEqual(observed[0]["reasoning_effort"], "high")
        self.assertFalse(observed[0]["ignore_eos"])

    def test_last_answer_and_paired_flips(self):
        cases = [{"id": i, "gold": "A", "prompt": "test"} for i in range(4)]
        candidate = {
            i: {
                "id": i,
                "gold": "A",
                "prediction": text,
                "output_tokens": 8192,
                "finish_reason": "length",
                "seconds": 100,
                "final_answer": None,
            }
            for i, text in enumerate(("Answer: B Answer: a", "Answer: A", "no answer", "Answer: B"))
        }
        baseline = {i: {**candidate[i], "prediction": "Answer: B"} for i in range(4)}
        result = EXPERIMENT.summarize(cases, candidate, baseline)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["correct_final"], 0)
        self.assertEqual(result["output_limit_hits"], 4)
        self.assertEqual(result["cap_without_answer"], 1)
        self.assertEqual(result["baseline_wrong_candidate_right"], 2)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 0.5)
        self.assertFalse(result["full_198_score"])

    def test_resume_refuses_changed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            EXPERIMENT.bind_manifest(manifest, {"config": "first"})
            EXPERIMENT.bind_manifest(manifest, {"config": "first"})
            with self.assertRaisesRegex(ValueError, "changed configuration"):
                EXPERIMENT.bind_manifest(manifest, {"config": "second"})

    def test_resume_retries_failure_but_retains_success(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "results.jsonl"
            ledger.write_text(
                "\n".join(
                    json.dumps(r)
                    for r in ({"id": 0, "success": False}, {"id": 0, "success": True}, {"id": 1, "success": False})
                )
                + "\n"
            )
            self.assertEqual(set(EXPERIMENT.read_results(ledger)), {0})
            with ledger.open("a") as output:
                output.write('{"id":')
            with self.assertRaises(json.JSONDecodeError):
                EXPERIMENT.read_results(ledger)

    def test_wrong_cases_and_wrong_prompt_digest_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = Path(directory) / "cases.json"
            cases.write_text("[]")
            with self.assertRaisesRegex(ValueError, "frozen"):
                EXPERIMENT.read_cases(cases)
        with self.assertRaisesRegex(ValueError, "prompt digest"):
            EXPERIMENT.check_rows(
                [{"id": 0, "gold": "A", "prompt": "test"}], {0: {"id": 0, "gold": "A", "prompt_sha256": "wrong"}}
            )

    def test_subset_comparison_accepts_full_baseline(self):
        cases = [{"id": 0, "gold": "A", "prompt": "test"}]
        row = {"id": 0, "gold": "A", "prediction": "Answer: A", "output_tokens": 1}
        baseline = {0: row, 1: {**row, "id": 1}}
        self.assertEqual(EXPERIMENT.summarize(cases, {0: row}, baseline)["paired_items"], 1)


if __name__ == "__main__":
    unittest.main()
