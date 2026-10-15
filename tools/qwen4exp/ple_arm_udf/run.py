# SPDX-License-Identifier: Apache-2.0
"""Run one ARM UDF permissions receipt in a private DataFlow session."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-config", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = args.output / "compile.json"
    config.write_text(
        json.dumps(
            {
                "func_list": [{"func_name": "PleAccess"}],
                "input_num": 1,
                "output_num": 1,
                "target_bin": "libple_access.so",
                "workspace": str(Path(__file__).resolve().parent),
                "cmakelist_path": "CMakeLists.txt",
            }
        )
    )
    deploy = args.output / "deploy.json"
    deploy.write_text(
        json.dumps(
            {
                "dynamic_schedule_enable": False,
                "batch_deploy_info": [{"flow_node_list": ["FlowNode"], "logic_device_list": "0:0:0:0"}],
            }
        )
    )
    # Runtime imports only in this explicitly invoked hardware diagnostic.
    import dataflow as df
    import numpy as np

    df.init(
        {
            "ge.exec.deviceId": "0",
            "ge.exec.logicalDeviceClusterDeployMode": "SINGLE",
            "ge.exec.logicalDeviceId": "[0:0]",
            "ge.resourceConfigPath": str(args.resource_config.resolve()),
            "ge.experiment.data_flow_deploy_info_path": str(deploy.resolve()),
            "ge.flowGraphMemMaxSize": "67108864",
        }
    )
    try:
        data = df.FlowData()
        point = df.FuncProcessPoint(compile_config_path=str(config))
        node = df.FlowNode(input_num=1, output_num=1)
        node.add_process_point(point)
        graph = df.FlowGraph([node(data)])
        status = graph.feed_data({data: np.array([1], dtype=np.int32)}, df.FlowInfo(), timeout=30000)
        if status != 0:
            raise RuntimeError(f"DataFlow feed failed: {status}")
        outputs, _info, status = graph.fetch_data(timeout=30000)
        if status != 0 or len(outputs) != 1:
            raise RuntimeError(f"DataFlow fetch failed: {status}, outputs={len(outputs)}")
        receipt = outputs[0].numpy().tobytes().split(b"\0", 1)[0].decode()
        (args.output / "receipt.txt").write_text(receipt)
        if "PLATFORM machine=aarch64" not in receipt or "DMA_SUBMITTED=0" not in receipt:
            raise RuntimeError("missing complete ARM process receipt")
        print(receipt)
    finally:
        df.finalize()


if __name__ == "__main__":
    main()
