import subprocess

from tools.glm_perf.resident_harness import ResidentClient

c = ResidentClient("http://127.0.0.1:8000", 6)
c.request("/resident/maintenance")
try:
    c.request("/pause?mode=wait&clear_cache=true")
    c.rpc("resident_reset")
    c.request("/resume")
    subprocess.run(
        [
            "python",
            "-m",
            "tools.minimax_m3.coherence_probe",
            "--extended",
            "--case",
            "greeting",
            "--case",
            "france",
            "--case",
            "sparse_retrieval",
            "--output",
            "/workspace/m31-batch128-baseline",
        ],
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "tools.minimax_m3.native_expert_benchmark",
            "/srv/ai/storage/artifacts/minimax-m31-20261024/expert0",
            "--manifest",
            "/vllm-workspace/minimax-native-v7/manifest.json",
            "--binding",
            "/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so",
            "--device",
            "npu:5",
            "--output",
            "/workspace/m31-native-v7-projections.json",
        ],
        check=True,
    )
finally:
    c.request("/resident/maintenance", method="DELETE")
