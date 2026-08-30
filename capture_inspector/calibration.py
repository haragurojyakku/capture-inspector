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

# The chart is square. A phone can then show it large in either orientation,
# instead of the user having to fight a 16:9 image into a portrait screen.
#
# Localisation borrows QR's finder pattern: concentric squares whose every
# scan line through the centre reads 1:1:3:1:1 dark-light-dark-light-dark.
# Being purely geometric it costs nothing when the colours are wrong, which
# matters on a chart whose whole job is to measure colour error.
MODULE = 0.015
FINDER_MODULES = 7          # QR finder: 7x7, ringed by a light quiet zone
ALIGN_MODULES = 5           # QR alignment: 5x5, deliberately smaller
QUIET_MODULES = 1

FINDER_RATIO = (1, 1, 3, 1, 1)
ALIGN_RATIO = (1, 1, 1, 1, 1)

# Centres sit the same distance from each corner so the four points form a
# square, which keeps the homography well conditioned.
CORNER_INSET = 0.02 + (FINDER_MODULES / 2) * MODULE

MARKER_POS: dict[str, tuple[float, float]] = {
    "tl": (CORNER_INSET, CORNER_INSET),
    "tr": (1 - CORNER_INSET, CORNER_INSET),
    "br": (1 - CORNER_INSET, 1 - CORNER_INSET),
    "bl": (CORNER_INSET, 1 - CORNER_INSET),
}
# Three finders and one smaller alignment pattern, exactly as a QR symbol does
# it: the odd corner out is what tells us which way up the chart is.
ALIGNMENT_CORNER = "br"

# Keep-out around each corner, so patches never touch a locator pattern.
CORNER_KEEPOUT = 0.02 + (FINDER_MODULES + QUIET_MODULES) * MODULE
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


def _row(group: str, entries, y: float, h: float, x0: float, x1: float) -> list[Patch]:
    """Lay `entries` out evenly between x0 and x1 at height `y`."""
    n = len(entries)
    span = x1 - x0
    gap = span * 0.008
    w = (span - gap * (n - 1)) / n
    return [
        Patch(name=name, group=group, x=x0 + i * (w + gap), y=y, w=w, h=h, ref=ref)
        for i, (name, ref) in enumerate(entries)
    ]


def build_layout() -> list[Patch]:
    """Ten rows on a square: two short ones tucked between the corner patterns,
    eight full-width ones in the band below them."""

    def gray(prefix: str, levels) -> list[tuple[str, tuple[int, int, int]]]:
        return [(f"{prefix}{v}", (v, v, v)) for v in levels]

    edge, inner_edge = 0.025, CORNER_KEEPOUT + 0.015
    band_top, band_bottom = CORNER_KEEPOUT + 0.012, 1 - CORNER_KEEPOUT - 0.012
    short_h = CORNER_KEEPOUT - edge - 0.012

    middle = [
        ("gray", gray("gray", GRAY_STEPS[:9])),
        ("gray", gray("gray", GRAY_STEPS[9:])),
        ("primary", PRIMARIES),
        ("hue_vivid", HUE_VIVID),
        ("hue_pastel", HUE_PASTEL),
        ("hue_deep", HUE_DEEP),
        ("mid", MID_PRIMARIES),
        ("memory", MEMORY),
    ]
    step = (band_bottom - band_top) / len(middle)

    patches = _row("black_end", gray("k", BLACK_END),
                   y=edge, h=short_h, x0=inner_edge, x1=1 - inner_edge)
    for i, (group, entries) in enumerate(middle):
        patches += _row(group, entries,
                        y=band_top + i * step, h=step * 0.9, x0=edge, x1=1 - edge)
    patches += _row("white_end", gray("w", WHITE_END),
                    y=1 - edge - short_h, h=short_h, x0=inner_edge, x1=1 - inner_edge)
    return patches


LAYOUT = build_layout()


# --------------------------------------------------------------- rendering
def render_pattern(side: int = 1080, height: int | None = None):
    """Draw the chart. The chart is square; `height` is accepted and ignored."""
    from PIL import Image, ImageDraw

    width = height = side
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for patch in LAYOUT:
        x0 = int(round(patch.x * width))
        y0 = int(round(patch.y * height))
        x1 = int(round((patch.x + patch.w) * width))
        y1 = int(round((patch.y + patch.h) * height))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=patch.ref)

    _draw_markers(draw, width, height)
    return img


def _draw_markers(draw, width: int, height: int) -> None:
    """QR-style locator patterns: three finders and one alignment pattern.

    Each is drawn on a light quiet zone. The chart's background is black, so
    without that quiet zone the outermost dark ring would merge into the
    background and the 1:1:3:1:1 scan signature would be lost.
    """
    module = MODULE * min(width, height)

    for name, (nx, ny) in MARKER_POS.items():
        cx, cy = nx * width, ny * height
        if name == ALIGNMENT_CORNER:
            rings = [(ALIGN_MODULES + 2 * QUIET_MODULES, (255, 255, 255)),
                     (ALIGN_MODULES, (0, 0, 0)),
                     (ALIGN_MODULES - 2, (255, 255, 255)),
                     (1, (0, 0, 0))]
        else:
            rings = [(FINDER_MODULES + 2 * QUIET_MODULES, (255, 255, 255)),
                     (FINDER_MODULES, (0, 0, 0)),
                     (FINDER_MODULES - 2, (255, 255, 255)),
                     (FINDER_MODULES - 4, (0, 0, 0))]

        for modules, colour in rings:
            half = modules * module / 2
            draw.rectangle(
                [int(round(cx - half)), int(round(cy - half)),
                 int(round(cx + half)) - 1, int(round(cy + half)) - 1],
                fill=colour,
            )


# ---------------------------------------------------------------- locating
@dataclass
class Localization:
    """Where the chart is in a captured frame, as chart coords -> pixel coords."""

    homography: np.ndarray
    method: str
    markers: dict[str, tuple[float, float]] | None = None

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        """Map normalised chart coordinates to pixel coordinates."""
        homogeneous = np.column_stack([pts, np.ones(len(pts))])
        out = homogeneous @ self.homography.T
        return out[:, :2] / out[:, 2:3]

    def chart_size(self) -> tuple[float, float]:
        """Width and height the chart occupies in the frame, in pixels."""
        corners = self.map_points(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
        top = np.linalg.norm(corners[1] - corners[0])
        bottom = np.linalg.norm(corners[2] - corners[3])
        left = np.linalg.norm(corners[3] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])
        return float((top + bottom) / 2), float((left + right) / 2)

    def sample_box(self) -> tuple[float, float]:
        """Pixels actually averaged for the tightest patch on the chart.

        This is the number that decides whether a reading means anything: the
        chart can be located perfectly and still be far too small to measure,
        because the display scaled it down and blended neighbouring patches
        into each other.
        """
        width, height = self.chart_size()
        narrowest = min(p.w for p in LAYOUT)
        shortest = min(p.h for p in LAYOUT)
        return (narrowest * width * SAMPLE_FRACTION,
                shortest * height * SAMPLE_FRACTION)


def _otsu(lum: np.ndarray) -> float:
    """Threshold that best separates the frame into dark and light."""
    hist, edges = np.histogram(lum, bins=64, range=(0, 255))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 128.0
    centres = (edges[:-1] + edges[1:]) / 2
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return 128.0
    cum = np.cumsum(hist * centres)
    mean_bg = np.divide(cum, weight_bg, out=np.zeros_like(cum), where=weight_bg > 0)
    mean_fg = np.divide(cum[-1] - cum, weight_fg, out=np.zeros_like(cum), where=weight_fg > 0)
    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    variance[~valid] = -1
    return float(centres[int(np.argmax(variance))])


def _runs(line: np.ndarray):
    """Run-length encode a boolean line into (value, start, length) arrays."""
    if line.size == 0:
        return np.array([]), np.array([]), np.array([])
    change = np.flatnonzero(np.diff(line)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [line.size]))
    return line[starts], starts, ends - starts


def _scan(line: np.ndarray, ratio, tolerance: float = 0.55):
    """Centres along one line whose run lengths match `ratio`, starting dark.

    This is QR's own trick: the finder is built so that any line through its
    middle crosses runs in a fixed proportion, whatever the angle or the scale.
    Matching proportions rather than sizes is what makes it scale-free.
    """
    values, starts, lengths = _runs(line)
    n = len(ratio)
    total_ratio = sum(ratio)
    out = []
    for i in range(len(values) - n + 1):
        if not values[i]:            # the pattern must begin on a dark run
            continue
        window = lengths[i:i + n]
        unit = window.sum() / total_ratio
        if unit < 0.8:
            continue
        if np.all(np.abs(window - np.array(ratio) * unit) <= tolerance * unit):
            middle = i + n // 2
            out.append((starts[middle] + lengths[middle] / 2, unit))
    return out


def _locator_centres(dark: np.ndarray, ratio, stride: int = 2):
    """Find locator patterns by scanning rows, then confirming down columns."""
    height, width = dark.shape
    found: list[tuple[float, float, float]] = []

    for y in range(0, height, stride):
        for cx, unit in _scan(dark[y], ratio):
            column = dark[:, int(round(cx))]
            for cy, unit_v in _scan(column, ratio):
                if abs(cy - y) > unit * 2:
                    continue
                # The horizontal and vertical module sizes only match when the
                # chart arrived square. Allow a wide mismatch so a display that
                # stretches the image unevenly still resolves - the run-ratio
                # test already validated the shape along each axis separately.
                if not 0.33 < unit_v / unit < 3.0:
                    continue
                found.append((cx, cy, (unit + unit_v) / 2))
                break

    # Every row crossing the same pattern reports it, so collapse the cluster.
    clusters: list[list[tuple[float, float, float]]] = []
    for cx, cy, unit in found:
        for group in clusters:
            gx, gy, gu = group[0]
            if abs(cx - gx) < gu * 3 and abs(cy - gy) < gu * 3:
                group.append((cx, cy, unit))
                break
        else:
            clusters.append([(cx, cy, unit)])

    return [
        (float(np.mean([c[0] for c in g])),
         float(np.mean([c[1] for c in g])),
         float(np.mean([c[2] for c in g])))
        for g in clusters if len(g) >= 2
    ]


CORNER_ORDER = ("tl", "tr", "br", "bl")


def _is_convex(pts: np.ndarray) -> bool:
    """True when the four points wind consistently, i.e. form a real quad."""
    signs = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        u, v = b - a, c - b
        signs.append(np.sign(u[0] * v[1] - u[1] * v[0]))
    return abs(sum(signs)) == 4


def _sample(frame: np.ndarray, loc: "Localization", names, grid: int = 5):
    """Mean RGB for named patches - the shared core of measuring and validating."""
    h, w = frame.shape[:2]
    steps = np.linspace(0.5 - SAMPLE_FRACTION / 2, 0.5 + SAMPLE_FRACTION / 2, grid)
    uu, vv = np.meshgrid(steps, steps)
    unit = np.column_stack([uu.ravel(), vv.ravel()])
    by_name = {p.name: p for p in LAYOUT}

    out: dict[str, np.ndarray] = {}
    for name in names:
        patch = by_name.get(name)
        if patch is None:
            continue
        pts = np.column_stack([
            patch.x + unit[:, 0] * patch.w,
            patch.y + unit[:, 1] * patch.h,
        ])
        mapped = np.round(loc.map_points(pts)).astype(int)
        xs, ys = mapped[:, 0], mapped[:, 1]
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if inside.sum() < grid:
            continue
        out[name] = frame[ys[inside], xs[inside]].astype(float).mean(axis=0)
    return out


def _score_fit(frame: np.ndarray, homography: np.ndarray) -> float | None:
    """Check a candidate mapping against patches whose content we already know.

    Four points always fit a homography exactly, so there is no reprojection
    error to look at. Instead the mapping has to explain the chart: the patches
    we called black must come back dark, the white ones bright, and each pure
    primary must still lead on its own channel. A mapping built from the wrong
    blobs fails this immediately.
    """
    loc = Localization(homography, "markers")
    probe = _sample(frame, loc, ("gray0", "black", "gray255", "white", "red", "green", "blue"))
    if len(probe) < 7:
        return None

    dark = (probe["gray0"].mean() + probe["black"].mean()) / 2
    light = (probe["gray255"].mean() + probe["white"].mean()) / 2
    if light - dark < 60:
        return None

    for name, channel in (("red", 0), ("green", 1), ("blue", 2)):
        if int(np.argmax(probe[name])) != channel:
            return None
    return float(light - dark)


def _assign_corners(finders, alignment):
    """Name the four points, using the odd one out to fix the orientation.

    The alignment pattern marks one corner; of the three finders the one
    furthest from it is the opposite corner, and the remaining two fall either
    side of that diagonal. Which side is which comes from the sign of the cross
    product, so a rotated or mirrored chart still resolves correctly.
    """
    br = np.array(alignment[:2])
    pts = [np.array(f[:2]) for f in finders]

    tl = max(pts, key=lambda p: np.hypot(*(p - br)))
    rest = [p for p in pts if p is not tl]
    if len(rest) != 2:
        return None

    diagonal = br - tl
    sides = [float(diagonal[0] * (p - tl)[1] - diagonal[1] * (p - tl)[0]) for p in rest]
    if sides[0] * sides[1] >= 0:      # both on one side: not a real corner set
        return None

    tr, bl = (rest[0], rest[1]) if sides[0] < 0 else (rest[1], rest[0])
    return {
        "tl": (float(tl[0]), float(tl[1])),
        "tr": (float(tr[0]), float(tr[1])),
        "br": (float(br[0]), float(br[1])),
        "bl": (float(bl[0]), float(bl[1])),
    }


def find_markers(frame: np.ndarray) -> dict[str, tuple[float, float]] | None:
    """Locate the QR-style locator patterns and return their pixel centres."""
    import itertools

    lum = frame.astype(np.float32).mean(axis=2)
    dark = lum < _otsu(lum)

    finders = _locator_centres(dark, FINDER_RATIO)
    alignments = _locator_centres(dark, ALIGN_RATIO)
    if len(finders) < 3 or not alignments:
        return None

    src = np.array([MARKER_POS[k] for k in CORNER_ORDER], dtype=float)
    best: tuple[float, dict[str, tuple[float, float]]] | None = None

    # A busy frame can throw up extra candidates, so try the plausible sets and
    # let the chart itself decide which one actually explains the picture.
    for trio in itertools.combinations(finders[:6], 3):
        units = [f[2] for f in trio]
        if max(units) / min(units) > 1.8:
            continue
        for align in alignments[:6]:
            corners = _assign_corners(trio, align)
            if corners is None:
                continue
            pts = np.array([corners[k] for k in CORNER_ORDER], dtype=float)
            if not _is_convex(pts):
                continue
            score = _score_fit(frame, solve_homography(src, pts))
            if score is not None and (best is None or score > best[0]):
                best = (score, corners)

    return best[1] if best is not None else None


def solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct linear transform for the 4-point chart -> frame mapping.

    A homography rather than an affine fit: it costs nothing extra with four
    correspondences and it also covers a chart photographed slightly off-axis,
    not just one scaled and letterboxed by a mirroring phone.
    """
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows, dtype=float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def locate(frame: np.ndarray) -> Localization | None:
    """Find the chart from its four locator patterns, or report failure.

    There is deliberately no fallback. The old one took the bounding box of
    everything bright, which only worked because the chart used to carry a
    white outer frame; against the square chart it would lock onto the patch
    grid and return a confidently wrong mapping. A clear failure the user can
    act on beats a measurement that is quietly off.
    """
    markers = find_markers(frame)
    if markers is None:
        return None
    src = np.array([MARKER_POS[k] for k in CORNER_ORDER], dtype=float)
    dst = np.array([markers[k] for k in CORNER_ORDER], dtype=float)
    return Localization(solve_homography(src, dst), "markers", markers)


def measure(frame: np.ndarray, loc: Localization, grid: int = 9) -> dict[str, np.ndarray]:
    """Mean RGB of every patch, sampled through the localisation.

    Sampling is done by mapping a grid of points from chart space into the
    frame, rather than by carving up a rectangle. That keeps the reading
    correct when the chart arrives rotated or slightly off-axis, where an
    axis-aligned crop would straddle patch boundaries.
    """
    return _sample(frame, loc, [p.name for p in LAYOUT], grid=grid)


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


# Below this many pixels per sampled patch, neighbouring patches have bled into
# each other badly enough that the numbers stop meaning anything.
SAMPLE_TOO_SMALL = 10.0
SAMPLE_MARGINAL = 20.0


def _judge_scale(report: ColorReport, loc: "Localization", frame_shape) -> None:
    """Warn when the chart was displayed too small to measure honestly."""
    box_w, box_h = loc.sample_box()
    smallest = min(box_w, box_h)
    chart_w, chart_h = loc.chart_size()
    coverage = (chart_w * chart_h) / (frame_shape[1] * frame_shape[0]) * 100

    detail = (
        f"チャートはフレームの {coverage:.1f}% "
        f"（{chart_w:.0f}x{chart_h:.0f} px）しか占めていません。\n"
        f"1パッチあたりの実測領域は約 {box_w:.0f}x{box_h:.0f} px です。"
    )
    fix = (
        "スマホを横向きにして、チャートを画面いっぱいに表示してください。\n"
        "写真ビューアで開いている場合は画像をタップしてUI（上部のファイル名や下部のツールバー）を隠すか、"
        "配布ページの「全画面で表示する」を使ってください。\n"
        "小さく表示すると表示側の縮小補間で隣のパッチの色が混ざり、測定値そのものが濁ります。"
    )

    if smallest < SAMPLE_TOO_SMALL:
        report.findings.insert(0, Finding(
            Severity.CRITICAL,
            "チャートが小さすぎます — この測定結果は使えません",
            detail + "\n\nこの大きさでは隣のパッチの色が混ざり込んでおり、"
                     "ここから作ったLUTは色被りを起こします。",
            fix,
        ))
    elif smallest < SAMPLE_MARGINAL:
        report.findings.insert(0, Finding(
            Severity.WARNING,
            "チャートが小さめです — 精度が落ちています",
            detail,
            fix,
        ))


def analyse_colors(
    measurements: dict[str, np.ndarray],
    loc: "Localization | None" = None,
    frame_shape=None,
) -> ColorReport:
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
    # Inserted at the front afterwards: if the chart was too small, that
    # outranks everything else below it, because it explains the rest.
    if loc is not None and frame_shape is not None:
        _judge_scale(report, loc, frame_shape)
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
