"""Measure the colour response of the whole capture chain, and correct it.

The idea: render a pattern of patches whose sRGB values we chose, display it on
the source device, capture it, and compare what came back against what we sent.
Everything between the two - the phone's HDMI output, the YUV encoding, the
card's conversion back to RGB - shows up as the difference.

Patch geometry lives in one place (`build_layout`) and is shared by the renderer
and the analyser, so the two can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .diagnose import Finding, Severity

# Fraction of the image taken by the white locator frame at the very edge.
BORDER = 0.014
# Only the middle of each patch is measured, so scaling softness at patch
# boundaries cannot contaminate the reading.
SAMPLE_FRACTION = 0.5

GRAY_STEPS = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
BLACK_END = [0, 4, 8, 12, 16, 20, 24, 32]
WHITE_END = [219, 227, 235, 240, 245, 250, 253, 255]

PRIMARIES = [
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
    ("cyan", (0, 255, 255)),
    ("magenta", (255, 0, 255)),
    ("yellow", (255, 255, 0)),
    ("white", (255, 255, 255)),
    ("black", (0, 0, 0)),
]

# Above this deviation from the identity, a 3x3 correction is worth the grid
# interpolation error that comes with moving from a 1D to a 3D LUT.
MATRIX_SIGNIFICANT = 0.02

HUE_COUNT = 12


def _hue_row(prefix: str, saturation: float, value: float) -> list[tuple[str, tuple[int, int, int]]]:
    """Evenly spaced hues at one saturation/brightness.

    A regular sweep beats a handful of hand-picked colours here: a matrix error
    rotates the whole hue circle, and that is far easier to see - and to fit -
    when the samples are evenly distributed around it.
    """
    import colorsys

    out = []
    for i in range(HUE_COUNT):
        h = i / HUE_COUNT
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        out.append((f"{prefix}{i * 360 // HUE_COUNT:03d}",
                    (round(r * 255), round(g * 255), round(b * 255))))
    return out


HUE_VIVID = _hue_row("hue", 1.0, 1.0)
HUE_PASTEL = _hue_row("pastel", 0.45, 1.0)
HUE_DEEP = _hue_row("deep", 1.0, 0.5)

# Rough stand-ins for familiar real-world colours. Not a licensed chart - they
# exist so hue errors show up somewhere the eye actually notices them.
MEMORY = [
    ("skin_pale", (238, 205, 186)),
    ("skin_light", (222, 170, 143)),
    ("skin_mid", (186, 133, 105)),
    ("skin_dark", (140, 96, 74)),
    ("skin_deep", (92, 62, 48)),
    ("sky_light", (140, 180, 220)),
    ("sky", (74, 120, 180)),
    ("sky_deep", (40, 72, 130)),
    ("foliage", (86, 122, 68)),
    ("grass", (120, 160, 70)),
    ("orange", (214, 126, 44)),
    ("brick", (150, 62, 48)),
]

# Half-intensity primaries, kept separate so the gamut corners are sampled at
# two brightnesses as well as at full scale.
MID_PRIMARIES = [
    ("red50", (128, 0, 0)),
    ("green50", (0, 128, 0)),
    ("blue50", (0, 0, 128)),
    ("cyan50", (0, 128, 128)),
    ("magenta50", (128, 0, 128)),
    ("yellow50", (128, 128, 0)),
    ("gray75", (191, 191, 191)),
    ("gray25", (64, 64, 64)),
    ("olive", (128, 128, 64)),
    ("teal", (64, 128, 128)),
    ("purple", (96, 64, 144)),
    ("rose", (200, 110, 140)),
]


@dataclass(frozen=True)
class Patch:
    """A rectangle in normalised image coordinates with the colour we sent."""

    name: str
    group: str
    x: float
    y: float
    w: float
    h: float
    ref: tuple[int, int, int]


def _row(group: str, entries, y: float, h: float, inner: float) -> list[Patch]:
    """Lay `entries` out evenly across the usable width at height `y`."""
    n = len(entries)
    gap = inner * 0.006
    total = inner - gap * (n - 1)
    w = total / n
    out: list[Patch] = []
    for i, (name, ref) in enumerate(entries):
        out.append(
            Patch(
                name=name,
                group=group,
                x=BORDER + i * (w + gap),
                y=y,
                w=w,
                h=h,
                ref=ref,
            )
        )
    return out


def build_layout() -> list[Patch]:
    inner = 1.0 - BORDER * 2
    patches: list[Patch] = []

    def gray(prefix: str, levels: list[int]):
        return [(f"{prefix}{v}", (v, v, v)) for v in levels]

    rows = [
        ("gray", gray("gray", GRAY_STEPS), 0.020, 0.112),
        ("black_end", gray("k", BLACK_END), 0.146, 0.072),
        ("white_end", gray("w", WHITE_END), 0.232, 0.072),
        ("primary", PRIMARIES, 0.318, 0.104),
        ("hue_vivid", HUE_VIVID, 0.436, 0.104),
        ("hue_pastel", HUE_PASTEL, 0.554, 0.104),
        ("hue_deep", HUE_DEEP, 0.672, 0.104),
        ("mid", MID_PRIMARIES, 0.790, 0.084),
        ("memory", MEMORY, 0.888, 0.092),
    ]
    for group, entries, y, h in rows:
        patches += _row(group, entries, y=BORDER + inner * y, h=inner * h, inner=inner)
    return patches


LAYOUT = build_layout()


# --------------------------------------------------------------- rendering
def render_pattern(width: int = 1920, height: int = 1080):
    """Draw the chart. Display this full-screen on the source device."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    bw = max(int(round(BORDER * min(width, height))), 2)
    draw.rectangle([bw, bw, width - bw - 1, height - bw - 1], fill=(0, 0, 0))

    for patch in LAYOUT:
        x0 = int(round(patch.x * width))
        y0 = int(round(patch.y * height))
        x1 = int(round((patch.x + patch.w) * width))
        y1 = int(round((patch.y + patch.h) * height))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=patch.ref)

    return img


# ---------------------------------------------------------------- locating
def find_pattern_bbox(frame: np.ndarray, threshold: int = 60) -> tuple[int, int, int, int] | None:
    """Find the chart inside a captured frame.

    The chart's outermost pixels are a white frame, and anything around it -
    letterbox bars from an aspect-ratio mismatch - is black. So the bounding box
    of everything bright enough is the chart itself. Because patch positions are
    then read proportionally inside that box, this also absorbs any stretching.
    """
    lum = frame.max(axis=2)
    mask = lum > threshold

    rows = np.where(mask.sum(axis=1) > mask.shape[1] * 0.5)[0]
    cols = np.where(mask.sum(axis=0) > mask.shape[0] * 0.5)[0]
    if rows.size < 2 or cols.size < 2:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def measure(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    """Mean RGB of every patch, read from the middle of each rectangle."""
    x0, y0, x1, y1 = bbox
    span_x, span_y = x1 - x0, y1 - y0
    out: dict[str, np.ndarray] = {}

    for patch in LAYOUT:
        inset_w = patch.w * (1 - SAMPLE_FRACTION) / 2
        inset_h = patch.h * (1 - SAMPLE_FRACTION) / 2
        px0 = x0 + int(round((patch.x + inset_w) * span_x))
        px1 = x0 + int(round((patch.x + patch.w - inset_w) * span_x))
        py0 = y0 + int(round((patch.y + inset_h) * span_y))
        py1 = y0 + int(round((patch.y + patch.h - inset_h) * span_y))
        if px1 <= px0 or py1 <= py0:
            continue
        region = frame[py0:py1, px0:px1].reshape(-1, 3)
        out[patch.name] = region.mean(axis=0)
    return out


# ---------------------------------------------------------------- analysis
@dataclass
class ColorReport:
    measurements: dict[str, np.ndarray]
    black_level: float = 0.0
    white_level: float = 0.0
    gamma: float = 1.0
    gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    black_clipped_below: int | None = None
    white_clipped_above: int | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def worst(self) -> Severity:
        for level in (Severity.CRITICAL, Severity.WARNING, Severity.INFO, Severity.OK):
            if any(f.severity is level for f in self.findings):
                return level
        return Severity.OK

    @property
    def looks_limited_range(self) -> bool:
        """Black near 16 and white near 235 means limited-range codes were
        handed over untouched as if they were full-range."""
        return self.black_level > 9 and self.white_level < 243


def _ramp(measurements: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    refs, meas = [], []
    for level in GRAY_STEPS:
        key = f"gray{level}"
        if key in measurements:
            refs.append(level)
            meas.append(measurements[key])
    return np.array(refs, dtype=float), np.array(meas, dtype=float)


def _first_distinct(levels: list[int], measurements: dict[str, np.ndarray], prefix: str, tol: float = 0.75):
    """Lowest level whose reading separates from its neighbour - i.e. where
    detail stops being crushed.

    The tolerance is deliberately tight: genuinely clipped patches read as the
    same value, whereas a steep gamma merely compresses them. Anything looser
    reports a gamma curve as lost detail.
    """
    vals = [(v, measurements[f"{prefix}{v}"].mean()) for v in levels if f"{prefix}{v}" in measurements]
    for (a, va), (b, vb) in zip(vals, vals[1:]):
        if abs(vb - va) > tol:
            return a
    return None


def _fit_channel(refs: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Fit `value = black + span * (ref/255)**gamma` for one channel.

    Black level, white level and gamma cannot be estimated independently: a
    straight-line fit reads a gamma curve as a shifted black point, and
    normalising by a wrong black point then distorts the gamma. So gamma is
    swept and the two linear terms solved exactly at each step, which turns the
    three-way fit into a one-dimensional search with a closed form inside it.
    """
    x = np.clip(refs / 255.0, 0.0, 1.0)
    best: tuple[float, float, float, float] | None = None

    for gamma in np.arange(0.40, 3.0005, 0.005):
        basis = np.column_stack([x**gamma, np.ones_like(x)])
        solution, *_ = np.linalg.lstsq(basis, values, rcond=None)
        residual = float(np.sum((basis @ solution - values) ** 2))
        if best is None or residual < best[0]:
            best = (residual, float(gamma), float(solution[0]), float(solution[1]))

    assert best is not None
    _, gamma, span, black = best
    return gamma, span, black


def analyse_colors(measurements: dict[str, np.ndarray]) -> ColorReport:
    report = ColorReport(measurements=measurements)
    add = report.findings.append

    refs, meas = _ramp(measurements)
    if refs.size < 4:
        add(Finding(Severity.CRITICAL, "測定できませんでした",
                    "グレースケールのパッチを読み取れませんでした。パターンが正しく全画面表示されているか確認してください。"))
        return report

    gammas, spans, blacks = [], [], []
    for c in range(3):
        gamma, span, black = _fit_channel(refs, meas[:, c])
        gammas.append(gamma)
        spans.append(span)
        blacks.append(black)

    report.gamma = float(np.mean(gammas))
    report.black_level = float(np.mean(blacks))
    report.white_level = float(np.mean(blacks) + np.mean(spans))

    # Relative channel gains: how unevenly the three channels use their range.
    span_arr = np.array(spans)
    report.gain = tuple(float(v) for v in span_arr / span_arr.mean())

    report.black_clipped_below = _first_distinct(BLACK_END, measurements, "k")
    report.white_clipped_above = _first_distinct(list(reversed(WHITE_END)), measurements, "w")

    _judge(report)
    return report


def _judge(report: ColorReport) -> None:
    add = report.findings.append

    # --- levels ------------------------------------------------------------
    if report.looks_limited_range:
        add(Finding(
            Severity.CRITICAL,
            "リミテッドレンジがそのまま出力されています",
            f"黒が {report.black_level:.1f}、白が {report.white_level:.1f} に位置しています"
            f"（本来は 0 と 255）。\n"
            "映像がリミテッドレンジ(16-235)のまま、フルレンジとして扱われている状態です。"
            "黒が浮き、白が沈み、コントラストが不足します。",
            "OBS のソースのプロパティで「色範囲」を『一部』(Partial/Limited) に設定してください。"
            "それだけで解決する場合、LUTは不要です。",
        ))
    elif report.black_level < -3 or report.white_level > 259:
        add(Finding(
            Severity.WARNING,
            "レンジが過剰に伸張されています",
            f"黒が {report.black_level:.1f}、白が {report.white_level:.1f} と範囲外に振れています。\n"
            "フルレンジの映像をさらにリミテッド→フル展開している可能性があります。",
            "OBS の「色範囲」を『全部』(Full) に設定してください。",
        ))
    else:
        add(Finding(
            Severity.OK,
            "黒レベル・白レベルは正常です",
            f"黒 {report.black_level:.1f} / 白 {report.white_level:.1f}（理想は 0 / 255）",
        ))

    # --- clipping ----------------------------------------------------------
    if report.black_clipped_below is not None and report.black_clipped_below > 0:
        add(Finding(
            Severity.WARNING,
            f"暗部が潰れています（{report.black_clipped_below} 以下が同じ値）",
            "黒側のテストパッチが区別できません。この階調は失われており、"
            "LUTでは復元できません。",
            "送出側かキャプチャ側のレンジ設定を見直してください。",
        ))
    if report.white_clipped_above is not None and report.white_clipped_above < 255:
        add(Finding(
            Severity.WARNING,
            f"明部が飛んでいます（{report.white_clipped_above} 以上が同じ値）",
            "白側のテストパッチが区別できません。この階調も復元できません。",
            "送出側かキャプチャ側のレンジ設定を見直してください。",
        ))

    # --- gamma -------------------------------------------------------------
    if abs(report.gamma - 1.0) > 0.08:
        direction = "暗く" if report.gamma > 1 else "明るく"
        add(Finding(
            Severity.WARNING,
            f"ガンマがずれています (実測 {report.gamma:.3f})",
            f"中間調が本来より{direction}なっています。1.000 が理想です。",
            "生成されるLUTで補正できます。",
        ))
    else:
        add(Finding(Severity.OK, f"ガンマは正常です ({report.gamma:.3f})", "中間調の明るさは適正です。"))

    # --- channel balance ---------------------------------------------------
    gain = np.array(report.gain)
    if gain.size == 3 and float(np.max(np.abs(gain - 1.0))) > 0.03:
        names = "RGB"
        detail = " / ".join(f"{names[i]} {gain[i]:.3f}" for i in range(3))
        add(Finding(
            Severity.WARNING,
            "チャンネルバランスがずれています",
            f"白パッチの channel gain: {detail}（すべて 1.000 が理想）\n"
            "色被り（ホワイトバランスのずれ）が出ています。",
            "生成されるLUTで補正できます。",
        ))
    else:
        add(Finding(Severity.OK, "チャンネルバランスは正常です",
                    "R/G/B のゲイン差は 3% 未満です。"))

    # --- hue / matrix ------------------------------------------------------
    bleed = _primary_bleed(report)
    if bleed is not None and bleed > 12:
        add(Finding(
            Severity.WARNING,
            f"原色に他チャンネルの混入があります (最大 {bleed:.1f})",
            "純粋な赤/緑/青のパッチに、本来ゼロであるべき他チャンネルの成分が出ています。\n"
            "YUV→RGB の変換係数 (BT.601 と BT.709) の不一致が典型的な原因です。",
            "OBS のソースのプロパティで「色空間」を 709 と 601 で切り替えて再測定してみてください。",
        ))


def _primary_bleed(report: ColorReport) -> float | None:
    """How much of the other two channels leaks into a pure primary."""
    worst = 0.0
    seen = False
    for i, name in enumerate(("red", "green", "blue")):
        m = report.measurements.get(name)
        if m is None:
            continue
        seen = True
        others = [m[j] for j in range(3) if j != i]
        worst = max(worst, max(others) - report.black_level)
    return worst if seen else None


# -------------------------------------------------------------- correction
def tone_curves(report: ColorReport, size: int) -> np.ndarray:
    """Per-channel inverse of the measured response, sampled on a uniform grid.

    The chart gives us reference -> measured for each channel. Applying the
    inverse to captured video puts the levels back where they were authored.
    Interpolation needs a monotonically increasing x, so the measured curve is
    forced monotonic first; flat (clipped) stretches simply stay flat, which is
    honest - the detail is gone and no curve can invent it.
    """
    refs, meas = _ramp(report.measurements)
    if refs.size < 4:
        raise ValueError("補正カーブを作るための測定値が足りません。")

    grid = np.linspace(0.0, 1.0, size)
    curves = []
    for c in range(3):
        x = meas[:, c] / 255.0
        y = refs / 255.0
        x = np.maximum.accumulate(x)  # enforce monotonic input for interp
        for i in range(1, x.size):
            if x[i] <= x[i - 1]:
                x[i] = x[i - 1] + 1e-6
        curves.append(np.clip(np.interp(grid, x, y), 0.0, 1.0))
    return np.array(curves)


def _apply_curves(values: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """Map values in [0,1] through the per-channel tone curves."""
    grid = np.linspace(0.0, 1.0, curves.shape[1])
    out = np.empty_like(values, dtype=float)
    for c in range(3):
        out[..., c] = np.interp(values[..., c], grid, curves[c])
    return out


def fit_matrix(report: ColorReport, curves: np.ndarray) -> np.ndarray:
    """Fit the 3x3 that remains after tone correction.

    A YUV matrix mismatch - encoding as BT.709 and decoding as BT.601, say - is
    a single linear transform of gamma-encoded RGB, and it leaves the neutral
    axis alone because both matrices' luma coefficients sum to one. So the grey
    ramp captures the tone error, and whatever is left over on the colour
    patches is this matrix. Fitting it needs colours spread around the hue
    circle, which is what the extra patches are for.
    """
    measured, target = [], []
    for patch in LAYOUT:
        if patch.group in ("gray", "black_end", "white_end"):
            continue
        m = report.measurements.get(patch.name)
        if m is None:
            continue
        # A channel sitting on 0 or 255 has been clamped, so it no longer says
        # where the transform would have put it. Fitting through those points
        # drags the matrix badly - fully saturated primaries are exactly the
        # patches a matrix error pushes out of gamut. The less saturated rows
        # survive this filter, which is what they are there for.
        if np.any(m <= 1.5) or np.any(m >= 253.5):
            continue
        measured.append(m / 255.0)
        target.append(np.array(patch.ref, dtype=float) / 255.0)

    if len(measured) < 6:
        return np.eye(3)

    corrected = _apply_curves(np.array(measured), curves)
    solution, *_ = np.linalg.lstsq(corrected, np.array(target), rcond=None)
    return solution.T  # rows act on a column vector


def matrix_strength(matrix: np.ndarray) -> float:
    """How far the fitted matrix is from doing nothing."""
    return float(np.abs(matrix - np.eye(3)).max())


def build_3d_lut(report: ColorReport, size: int = 17, curve_size: int = 64) -> str:
    """A .cube 3D LUT combining the tone curves with the fitted matrix."""
    curves = tone_curves(report, curve_size)
    matrix = fit_matrix(report, curves)

    axis = np.linspace(0.0, 1.0, size)
    # .cube 3D ordering: the red index moves fastest, blue slowest.
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([r, g, b], axis=-1).reshape(-1, 3)

    out = np.clip(_apply_curves(grid, curves) @ matrix.T, 0.0, 1.0)

    lines = [
        "# CaptureInspector color correction (tone curves + 3x3 matrix)",
        f"# black={report.black_level:.2f} white={report.white_level:.2f} gamma={report.gamma:.4f}",
        f"# matrix deviation={matrix_strength(matrix):.4f}",
        f"LUT_3D_SIZE {size}",
    ]
    lines += [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in out]
    return "\n".join(lines) + "\n"


def build_1d_lut(report: ColorReport, size: int = 64) -> str:
    """A .cube 1D LUT that undoes the measured per-channel response.

    Correct when the only errors are level and gamma. It cannot touch hue, so
    use `build_3d_lut` when a matrix error is present.
    """
    channels = tone_curves(report, size)

    lines = [
        "# CaptureInspector color correction",
        f"# black={report.black_level:.2f} white={report.white_level:.2f} gamma={report.gamma:.4f}",
        f"LUT_1D_SIZE {size}",
    ]
    for i in range(size):
        lines.append(f"{channels[0][i]:.6f} {channels[1][i]:.6f} {channels[2][i]:.6f}")
    return "\n".join(lines) + "\n"
