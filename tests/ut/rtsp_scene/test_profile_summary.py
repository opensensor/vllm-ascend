import unittest

from tools.rtsp_scene.profile_summary import interval_union, is_single_row_qkv, summarize


def kernel(start, duration, shape, device="1", kind="MatMulV2"):
    return {
        "Device_id": device,
        "Type": kind,
        "Output Shapes": shape,
        "Start Time(us)": str(start),
        "Duration(us)": str(duration),
    }


class ProfileSummaryTest(unittest.TestCase):
    def test_nz_qkv_requires_a_single_activation_scale_row(self):
        row = kernel(0, 10, '"192,1,16,16"', kind="QuantBatchMatmulV3")
        row["Input Shapes"] = '"128,1,16,32;128,192,16,32;3072;1"'
        self.assertTrue(is_single_row_qkv(row, 3072))
        row["Input Shapes"] = '"128,1,16,32;128,192,16,32;3072;16"'
        self.assertFalse(is_single_row_qkv(row, 3072))
        del row["Input Shapes"]
        self.assertFalse(is_single_row_qkv(row, 3072))

    def test_tensor_parallel_int8_qkv_markers_use_local_width(self):
        rows = [
            kernel(0, 50, "1024,3072", kind="QuantBatchMatmulV3"),
            kernel(50, 10, "1,3072", kind="QuantBatchMatmulV3"),
            kernel(65, 10, "1,3072", kind="QuantBatchMatmulV3"),
        ]
        result = summarize(rows, qkv_width=3072, layers=2)
        self.assertEqual(result["captured_decode_forwards"], 1)
        self.assertEqual(result["kernels"], 2)
        self.assertEqual(result["top_kernel_types"][0]["type"], "QuantBatchMatmulV3")

    def test_overlapping_tasks_are_not_counted_twice_in_device_activity(self):
        self.assertEqual(interval_union([(5, 8), (1, 6), (10, 12)]), 9)
        self.assertEqual(interval_union([]), 0)

    def test_shape_markers_exclude_vision_and_prefill(self):
        rows = [
            kernel(0, 100, "3744,1152"),
            kernel(100, 100, "1024,6144"),
            kernel(200, 10, '"1,6144"'),
            kernel(205, 10, '"1,6144"'),
            kernel(220, 5, "1,4096", kind="RmsNorm"),
        ]
        result = summarize(rows, layers=2)
        self.assertEqual(result["captured_decode_forwards"], 1)
        self.assertEqual(result["kernels"], 3)
        self.assertAlmostEqual(result["recorded_device_active_union_ms"], 0.020)
        self.assertAlmostEqual(result["recorded_device_gaps_ms"], 0.005)
        self.assertTrue(result["profiled_not_throughput_baseline"])

    def test_incomplete_layer_groups_and_multiple_devices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete M=1"):
            summarize([kernel(0, 10, "1,6144")])
        with self.assertRaisesRegex(ValueError, "exactly one device"):
            summarize([kernel(0, 10, "1,6144"), kernel(10, 10, "1,6144", device="0")], layers=2)
