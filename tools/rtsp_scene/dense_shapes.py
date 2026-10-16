"""Qwen3-VL dense projection geometry and 310P physical layouts."""

PROJECTIONS = ("qkv", "output", "gate_up", "down")
NZ_ROW_BLOCK = 16
NZ_INT8_COLUMN_BLOCK = 32
DENSE_OUTPUT_TILES = (64, 128, 256, 512)
DENSE_INNER_TILES = (128, 256, 512)
L0B_BYTES = 65536
MAX_CUBE_ROWS = 16
LEGACY_DECODE_ROWS = 8
LEGACY_NAMESPACE = "rtsp_dense_v1"


def kernel_namespace(build: dict) -> str:
    namespace = build.get("namespace", LEGACY_NAMESPACE)
    if (
        not isinstance(namespace, str)
        or not namespace.startswith("rtsp_dense_")
        or namespace == "rtsp_dense_"
        or not namespace.isascii()
        or not namespace.isidentifier()
    ):
        raise ValueError("Use an rtsp_dense_ native namespace with identifier characters")
    return namespace


def decode_rows(build: dict) -> int:
    rows = build.get("max_decode_rows", LEGACY_DECODE_ROWS)
    if type(rows) is not int or rows not in (LEGACY_DECODE_ROWS, MAX_CUBE_ROWS):
        raise ValueError("The frozen activation row limit must be 8 or 16")
    return rows


def validate_cube_tile(columns: int, inner: int) -> None:
    if columns not in DENSE_OUTPUT_TILES or inner not in DENSE_INNER_TILES:
        raise ValueError("Unsupported frozen Cube tile")
    if columns * inner > L0B_BYTES:
        raise ValueError("The output/inner tile must fit the 64 KiB L0B")


def shard_slices(projection: str, rows: int, columns: int, rank: int, size: int) -> tuple[slice, slice]:
    if projection not in PROJECTIONS or min(rows, columns, size) <= 0 or not 0 <= rank < size:
        raise ValueError("Invalid projection or tensor parallel geometry")
    dimension = rows if projection in ("qkv", "gate_up") else columns
    if dimension % size:
        raise ValueError("Projection dimension must divide evenly across tensor parallel ranks")
    local = slice(rank * dimension // size, (rank + 1) * dimension // size)
    return (local, slice(None)) if projection in ("qkv", "gate_up") else (slice(None), local)


def nz_shape(rows: int, columns: int, int8: bool = False) -> tuple[int, int, int, int]:
    if min(rows, columns) <= 0:
        raise ValueError("NZ dimensions must be positive")
    block = NZ_INT8_COLUMN_BLOCK if int8 else NZ_ROW_BLOCK
    return (
        (columns + block - 1) // block,
        (rows + NZ_ROW_BLOCK - 1) // NZ_ROW_BLOCK,
        NZ_ROW_BLOCK,
        block,
    )
