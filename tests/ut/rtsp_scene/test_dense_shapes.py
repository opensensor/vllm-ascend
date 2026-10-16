import unittest

from tools.rtsp_scene.dense_shapes import decode_rows, kernel_namespace, nz_shape, shard_slices, validate_cube_tile


class DenseShapesTest(unittest.TestCase):
    def test_frozen_build_defaults_and_limits(self):
        self.assertEqual(decode_rows({}), 8)
        self.assertEqual(kernel_namespace({}), "rtsp_dense_v1")
        self.assertEqual(decode_rows({"max_decode_rows": 16}), 16)
        self.assertEqual(kernel_namespace({"namespace": "rtsp_dense_scatter16"}), "rtsp_dense_scatter16")
        for rows in (0, 9, 17, True, "16"):
            with self.assertRaises(ValueError):
                decode_rows({"max_decode_rows": rows})
        for namespace in (None, "", "other_library", "rtsp_dense_x;command", "rtsp_dense_x-y"):
            with self.assertRaises(ValueError):
                kernel_namespace({"namespace": namespace})

    def test_cube_tiles_respect_l0b_capacity(self):
        for columns, inner in ((128, 512), (256, 256), (512, 128)):
            validate_cube_tile(columns, inner)
        for columns, inner in ((512, 512), (256, 512), (511, 128), (128, 32)):
            with self.assertRaises(ValueError):
                validate_cube_tile(columns, inner)

    def test_column_shards_slice_each_qkv_or_gate_piece_before_concatenation(self):
        self.assertEqual(shard_slices("qkv", 1024, 4096, 1, 2), (slice(512, 1024), slice(None)))
        self.assertEqual(shard_slices("gate_up", 12288, 4096, 0, 2), (slice(0, 6144), slice(None)))

    def test_row_shards_retain_full_per_channel_scales(self):
        self.assertEqual(shard_slices("down", 4096, 12288, 1, 2), (slice(None), slice(6144, 12288)))
        self.assertEqual(shard_slices("output", 4096, 4096, 0, 2), (slice(None), slice(0, 2048)))

    def test_nz_layout_accounts_for_dtype_and_padding(self):
        self.assertEqual(nz_shape(1, 4096, True), (128, 1, 16, 32))
        self.assertEqual(nz_shape(12288, 4096, True), (128, 768, 16, 32))
        self.assertEqual(nz_shape(8, 12288), (768, 1, 16, 16))
        self.assertEqual(nz_shape(17, 33, True), (2, 2, 16, 32))

    def test_invalid_geometry_is_rejected(self):
        for arguments in (("head", 32, 32, 0, 2), ("qkv", 31, 32, 0, 2), ("down", 32, 32, 2, 2)):
            with self.assertRaises(ValueError):
                shard_slices(*arguments)
        with self.assertRaises(ValueError):
            nz_shape(0, 32)
