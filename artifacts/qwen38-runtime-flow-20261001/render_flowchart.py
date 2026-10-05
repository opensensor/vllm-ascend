#!/usr/bin/env python3
"""Render the retained Qwen3.8 Flash Next 310P execution flow."""

# ruff: noqa: E501  # Diagram labels and their coordinates stay together.

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).with_name("qwen38-prefill-decode-flow.png")
WIDTH, HEIGHT = 2800, 2200

BG = "#F7F9FC"
INK = "#172033"
MUTED = "#526077"
HOST = "#FFF1CE"
NPU = "#E8F2FF"
COMM = "#F0E8FF"
CACHE = "#E8F7EE"
GRAPH = "#E8F8F6"
HOT = "#C43C3C"
BLUE = "#2868B2"
GREEN = "#247A57"
PURPLE = "#7652A7"
LINE = "#7F8BA0"
WHITE = "#FFFFFF"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


image = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(image)


def wrap(text: str, max_width: int, text_font: ImageFont.FreeTypeFont) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split()
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if draw.textbbox((0, 0), candidate, font=text_font)[2] <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def text_block(
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    *,
    fill: str,
    outline: str = LINE,
    title_color: str = INK,
    body_color: str = MUTED,
    title_size: int = 31,
    body_size: int = 25,
    radius: int = 20,
    width: int = 3,
    pad: int = 24,
) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    tf = font(title_size, True)
    bf = font(body_size)
    draw.text((x1 + pad, y1 + pad), title, font=tf, fill=title_color)
    y = y1 + pad + title_size + 14
    for line in wrap(body, x2 - x1 - 2 * pad, bf):
        draw.text((x1 + pad, y), line, font=bf, fill=body_color)
        y += body_size + 8


def centered_label(xy: tuple[int, int, int, int], text: str, fill: str, color: str = WHITE) -> None:
    draw.rounded_rectangle(xy, radius=20, fill=fill)
    f = font(34, True)
    box = draw.textbbox((0, 0), text, font=f)
    x1, y1, x2, y2 = xy
    draw.text(((x1 + x2 - (box[2] - box[0])) / 2, (y1 + y2 - (box[3] - box[1])) / 2 - 3), text, font=f, fill=color)


def arrow(start: tuple[int, int], end: tuple[int, int], *, color: str = LINE, width: int = 6) -> None:
    draw.line((start, end), fill=color, width=width)
    x2, y2 = end
    x1, y1 = start
    if abs(y2 - y1) >= abs(x2 - x1):
        direction = 1 if y2 > y1 else -1
        points = [(x2, y2), (x2 - 14, y2 - 22 * direction), (x2 + 14, y2 - 22 * direction)]
    else:
        direction = 1 if x2 > x1 else -1
        points = [(x2, y2), (x2 - 22 * direction, y2 - 14), (x2 - 22 * direction, y2 + 14)]
    draw.polygon(points, fill=color)


def pill(x: int, y: int, text: str, fill: str, text_color: str = INK) -> int:
    f = font(22, True)
    box = draw.textbbox((0, 0), text, font=f)
    w = box[2] - box[0] + 34
    draw.rounded_rectangle((x, y, x + w, y + 42), radius=21, fill=fill)
    draw.text((x + 17, y + 8), text, font=f, fill=text_color)
    return x + w


# Title and legend.
draw.text((60, 40), "Qwen3.8 Flash Next — retained 4× Ascend 310P execution flow", font=font(50, True), fill=INK)
draw.text(
    (62, 105),
    "Native INT4 W4A8 experts • TP4 / EP4 • MTP2 • 262K maximum context • scheduler chunk = 512 tokens",
    font=font(27),
    fill=MUTED,
)

legend_x = 1850
legend_y = 158
for label, color in (("Host / scheduler", HOST), ("NPU compute", NPU), ("Collective", COMM), ("Cache / state", CACHE)):
    draw.rounded_rectangle(
        (legend_x, legend_y, legend_x + 34, legend_y + 34), radius=8, fill=color, outline=LINE, width=2
    )
    draw.text((legend_x + 48, legend_y + 2), label, font=font(22), fill=MUTED)
    legend_x += 230

text_block(
    (300, 215, 2500, 340),
    "OpenAI request → tokenizer → vLLM scheduler",
    "CPU builds request metadata, block tables, positions, and the token batch. The execution path then depends on whether tokens are prompt prefill or iterative decode.",
    fill=HOST,
    title_size=32,
    body_size=24,
)

# Split arrows and lane headers.
arrow((1400, 340), (700, 405), color=LINE)
arrow((1400, 340), (2100, 405), color=LINE)
centered_label((60, 395, 1340, 475), "COLD / CHUNKED PREFILL", "#B66A1F")
centered_label((1460, 395, 2740, 475), "STEADY DECODE + MTP", BLUE)

# Prefill lane.
text_block(
    (60, 505, 1340, 625),
    "1. Schedule one 512-token prompt chunk",
    "The full prompt is not executed at once. Every chunk traverses all 48 layers before the next chunk advances.",
    fill=HOST,
)
arrow((700, 625), (700, 655))

text_block(
    (60, 655, 1340, 820),
    "2. PLE injection at layer 2  ⚠",
    "Host n-gram hashing and row lookup → selected FP16 rows from the lazy/shared host table → host-to-device staging → W8A8 2560→12800 projection → gate + dilated short convolution.",
    fill=HOST,
    outline=HOT,
    width=5,
)
arrow((700, 820), (700, 850))

draw.rounded_rectangle((60, 850, 1340, 1535), radius=24, fill=WHITE, outline=BLUE, width=4)
draw.text((88, 875), "3. Run the 48-layer stack for this chunk", font=font(32, True), fill=INK)
draw.text(
    (88, 920),
    "36 GDN layers + 12 QSA layers (every fourth layer) • 4 hyper-connection streams",
    font=font(23),
    fill=MUTED,
)

text_block(
    (90, 970, 690, 1165),
    "GDN attention ×36",
    "Project Q/K/V + gates; AscendC chunk-gated delta rule uses 64-token internal chunks; carry recurrent and convolution state across scheduler chunks.",
    fill=NPU,
    title_size=27,
    body_size=22,
)
text_block(
    (710, 970, 1310, 1165),
    "QSA attention ×12  ⚠",
    "Project Q/K/V/gate + index Q/K; compress/index/top-k; select ≤2,048 old tokens; retained per-query batched K/V gather (tile 64); sparse GQA + output gate.",
    fill=NPU,
    outline=HOT,
    width=5,
    title_size=27,
    body_size=22,
)
arrow((390, 1165), (390, 1200), color=BLUE)
arrow((1010, 1165), (1010, 1200), color=BLUE)

text_block(
    (90, 1200, 1310, 1455),
    "MoE in every layer: top-10 of 512 experts  ⚠",
    "Router → device dispatch/sort → EP4 ownership (128 experts per rank) → activation INT8 per-group quantize + INT4 pack on NPU → grouped gate/up, SwiGLU, grouped down projection → route-weight combine. HCCL all-to-all / reductions exchange routed work and TP outputs.",
    fill=COMM,
    outline=HOT,
    width=5,
    title_size=28,
    body_size=22,
)
draw.text((90, 1480), "Residual / RMSNorm / hyper-connection combine", font=font(23, True), fill=MUTED)

arrow((700, 1535), (700, 1565))
text_block(
    (60, 1565, 1340, 1685),
    "4. Persist state and cache writes",
    "Write GDN recurrent state, QSA ring/compressed side caches, and paged K/V. Then update block tables and metadata for the next scheduler step.",
    fill=CACHE,
)
arrow((700, 1685), (700, 1715))
text_block(
    (60, 1715, 1340, 1815),
    "5. Repeat until the entire prompt is consumed",
    "40K tokens ≈ 79 scheduler chunks × the complete layer stack. First output waits for all of them.",
    fill=HOST,
    outline=HOT,
    width=5,
    title_size=28,
    body_size=23,
)

# Decode lane.
text_block(
    (1460, 505, 2740, 625),
    "1. Schedule active requests + two MTP draft tokens",
    "One target token plus up to two speculative tokens per request; c1 and c4 select different qualified expert schedules.",
    fill=HOST,
)
arrow((2100, 625), (2100, 655))

text_block(
    (1460, 655, 2740, 820),
    "2. Replay the full decode ACL graph",
    "Captured batch sizes 1, 2, 3, and 6 cover target/MTP shapes. Static buffers and device metadata avoid rebuilding the Python operator graph on every token.",
    fill=GRAPH,
    outline=GREEN,
    width=5,
)
arrow((2100, 820), (2100, 850))

draw.rounded_rectangle((1460, 850, 2740, 1535), radius=24, fill=WHITE, outline=BLUE, width=4)
draw.text((1488, 875), "3. Run the 48-layer stack for this decode step", font=font(32, True), fill=INK)
draw.text(
    (1488, 920), "Small token batch; old context is read from persistent device caches", font=font(23), fill=MUTED
)

text_block(
    (1490, 970, 2090, 1165),
    "GDN recurrent ×36",
    "One-step recurrent kernel reads and updates compact GDN state instead of replaying the whole prompt.",
    fill=NPU,
    title_size=27,
    body_size=22,
)
text_block(
    (2110, 970, 2710, 1165),
    "QSA sparse lookup ×12",
    "Indexer chooses ≤2,048 relevant old tokens from QSA caches; gather selected K/V and run sparse GQA. Cost grows far less than full attention.",
    fill=NPU,
    title_size=27,
    body_size=22,
)
arrow((1790, 1165), (1790, 1200), color=BLUE)
arrow((2410, 1165), (2410, 1200), color=BLUE)

text_block(
    (1490, 1200, 2710, 1455),
    "Native INT4 MoE + collectives",
    "c1: streamed-weight dispatch schedule for ≤30 routed rows. c2–c4: resident-weight grouped schedule. Packed W4 weights remain INT4; activations quantize/pack on device. Current residual costs include dispatch/sort, HCCL waits, and synchronization around dependent work.",
    fill=COMM,
    title_size=28,
    body_size=22,
)
draw.text((1490, 1480), "Residual / RMSNorm / hyper-connection combine", font=font(23, True), fill=MUTED)

arrow((2100, 1535), (2100, 1565))
text_block(
    (1460, 1565, 2740, 1685),
    "4. LM head + MTP verification",
    "W8A8 dynamic LM head computes logits; rejection sampler verifies draft tokens, accepts 1–3 tokens, and updates all caches/states.",
    fill=NPU,
)
arrow((2100, 1685), (2100, 1715))
text_block(
    (1460, 1715, 2740, 1815),
    "5. Stream accepted tokens and replay",
    "Measured retained runtime: c1 29.59 tok/s; c4 60.24 tok/s aggregate (15.85 median per stream).",
    fill=GRAPH,
    outline=GREEN,
    width=5,
    title_size=28,
    body_size=23,
)

# Bottom explanation strip.
draw.rounded_rectangle((60, 1855, 2740, 2140), radius=24, fill="#FFF7F7", outline=HOT, width=4)
draw.text((90, 1880), "Why cold prefill is still the dominant problem", font=font(34, True), fill=HOT)
body_font = font(25)
left_notes = (
    "• 512-token scheduling repeats the full 48-layer orchestration ~79 times for a 40K prompt.\n"
    "• Prefill is eager/dynamic; FULL_DECODE_ONLY graphs accelerate decode, not this path.\n"
    "• PLE hashing/host row access/H2D and the 2560→12800 projection occur before layer 2."
)
right_notes = (
    "• QSA still plans and gathers selected K/V per query tile; reuse across neighboring queries is limited.\n"
    "• MoE dispatch/sort plus EP/TP collectives repeat in all 48 layers.\n"
    "• Retained cold 40K TTFT ≈130.4 s; group-major experiment regressed to 204–205 s."
)
for x, notes in ((100, left_notes), (1420, right_notes)):
    y = 1935
    for line in notes.split("\n"):
        for wrapped in wrap(line, 1220, body_font):
            draw.text((x, y), wrapped, font=body_font, fill=INK)
            y += 36
        y += 6

# Small configuration pills.
x = 70
for label, color in (
    ("W4 weights stay packed", "#DDEBFF"),
    ("A8 per-group activations", "#DDEBFF"),
    ("QSA budget 2,048", "#E6F5EC"),
    ("512-token scheduler chunks", "#FFE8D1"),
):
    x = pill(x, 2150, label, color) + 16

image.save(OUT, optimize=True)
print(OUT)
