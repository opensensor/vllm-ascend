import csv
import importlib.util
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve().parents[3] / "artifacts/qwen38-1m/run-gpqa-loopback.py"
SPEC = importlib.util.spec_from_file_location("gpqa_loopback", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class GPQALoopbackTest(unittest.TestCase):
    def test_prepare_cases_matches_aisbench_option_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "gpqa.csv"
            with dataset.open("w", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow([""] * 7 + ["Question", "Correct", "Wrong1", "Wrong2", "Wrong3"])
                writer.writerow([""] * 7 + ["Question one", "Correct", "Wrong1", "Wrong2", "Wrong3"])
                writer.writerow([""] * 7 + ["Question two", "Correct", "Wrong1", "Wrong2", "Wrong3"])

            cases = RUNNER.prepare_cases(dataset)
            self.assertEqual([case["gold"] for case in cases], ["D", "C"])
            self.assertIn("A) Wrong1\nB) Wrong2\nC) Wrong3\nD) Correct", cases[0]["prompt"])

    def test_replay_saves_and_resumes_without_repeat_requests(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(request)
                response = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"reasoning": "Thinking", "content": "Answer: B"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"completion_tokens": 5},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                results = Path(directory) / "results.jsonl"
                cases = [
                    {"id": 0, "prompt": "Question zero", "gold": "B"},
                    {"id": 1, "prompt": "Question one", "gold": "A"},
                ]
                url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
                RUNNER.run(cases, results, url, "test-model", 2, 5)
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0]["max_tokens"], 8192)
                self.assertEqual(requests[0]["seed"], 1024)
                self.assertEqual(RUNNER.load_successes(results)[0]["answer"], "B")

                RUNNER.run(cases, results, url, "test-model", 2, 5)
                self.assertEqual(len(requests), 2)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
