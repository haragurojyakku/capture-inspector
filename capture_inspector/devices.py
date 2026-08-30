"""Enumerate DirectShow capture devices and the video formats they advertise.

This is the same list OBS reads when it fills in its resolution/FPS dropdowns,
so if a resolution is missing here it will be missing in OBS too - which makes
this the right place to look when OBS "only offers the device default".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pygrabber.dshow_graph import FilterGraph

from . import usb_topology as ut

# Every FourCC media subtype embeds its four characters in the GUID's first
# field, e.g. {32595559-...} is 'YUY2' read little-endian.
_FOURCC_GUID = re.compile(r"^\{([0-9A-Fa-f]{8})-0000-0010-8000-00AA00389B71\}$")


@dataclass(frozen=True)
class VideoFormat:
    index: int
    width: int
    height: int
    fps: float
    subtype: str

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def __str__(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps:g}fps ({self.subtype})"


@dataclass
class CaptureDevice:
    index: int
    name: str
    formats: list[VideoFormat]
    current: tuple[int, int] | None = None
    topology: ut.Topology | None = None
    driver_service: str = ""
    instance_id: str = ""
    error: str = ""

    @property
    def resolutions(self) -> list[tuple[int, int]]:
        return sorted({f.resolution for f in self.formats}, key=lambda r: r[0] * r[1])

    @property
    def subtypes(self) -> list[str]:
        return sorted({f.subtype for f in self.formats})

    @property
    def best(self) -> VideoFormat | None:
        """Highest resolution, breaking ties on fps - the card's headline capability."""
        if not self.formats:
            return None
        return max(self.formats, key=lambda f: (f.pixels, f.fps))

    @property
    def max_fps(self) -> float:
        return max((f.fps for f in self.formats), default=0.0)

    def fps_for(self, resolution: tuple[int, int]) -> list[float]:
        return sorted({f.fps for f in self.formats if f.resolution == resolution})

    @property
    def is_virtual(self) -> bool:
        return "virtual" in self.name.lower()


def _subtype_name(guid: str) -> str:
    """Readable name for a media subtype, decoding FourCC GUIDs directly.

    pygrabber looks subtypes up in a fixed table and raises KeyError on
    anything missing from it - OBS's virtual camera emits I420, which is not
    there. Decoding the FourCC out of the GUID works for any of them, so the
    table becomes a nicety rather than a dependency.
    """
    from pygrabber.dshow_graph import subtypes

    known = subtypes.get(guid)
    if known:
        return known

    match = _FOURCC_GUID.match(guid)
    if match:
        text = int(match.group(1), 16).to_bytes(4, "little").decode("ascii", "replace")
        text = text.strip("\x00 ")
        if text and text.isprintable():
            return text
    return guid


def _enumerate_formats(device) -> list[VideoFormat]:
    """Video formats a device advertises, read straight off IAMStreamConfig."""
    from ctypes import POINTER, cast

    from comtypes import GUID
    from pygrabber.dshow_graph import VIDEOINFOHEADER, FormatTypes, IAMStreamConfig

    config = device.get_out().QueryInterface(IAMStreamConfig)
    count, _ = config.GetNumberOfCapabilities()
    video_info = GUID(FormatTypes.FORMAT_VideoInfo)

    formats: list[VideoFormat] = []
    for i in range(count):
        media_type, caps = config.GetStreamCaps(i)
        if video_info != media_type.contents.formattype:
            continue
        header = cast(media_type.contents.pbFormat, POINTER(VIDEOINFOHEADER)).contents.bmi_header
        # MinFrameInterval is the shortest gap between frames, so it gives the
        # highest rate - pygrabber's field naming has this back to front.
        interval = caps.MinFrameInterval
        formats.append(
            VideoFormat(
                index=i,
                width=header.biWidth,
                height=abs(header.biHeight),
                fps=round(10_000_000 / interval, 3) if interval else 0.0,
                subtype=_subtype_name(str(media_type.contents.subtype)),
            )
        )
    return formats


def _read_formats(graph_index: int) -> tuple[list[VideoFormat], tuple[int, int] | None]:
    graph = FilterGraph()
    graph.add_video_input_device(graph_index)
    device = graph.get_input_device()

    try:
        current = device.get_current_format()
    except Exception:  # noqa: BLE001 - some virtual cams refuse this
        current = None

    return _enumerate_formats(device), current


# Formats DirectShow can reliably bridge to the RGB24 the sample grabber wants.
_PREFERRED_SUBTYPES = ("YUY2", "NV12", "I420", "RGB24", "ARGB32")


def _grab_once(device_index: int, format_index: int | None, timeout: float):
    """One attempt at a still, optionally pinning the device's output format."""
    import threading
    import time

    from pygrabber.dshow_graph import FilterGraph

    holder: dict[str, object] = {}
    arrived = threading.Event()

    def on_frame(image):
        if not arrived.is_set():
            holder["frame"] = image[:, :, ::-1].copy()  # DirectShow RGB24 is BGR
            arrived.set()

    graph = None
    try:
        graph = FilterGraph()
        graph.add_video_input_device(device_index)
        if format_index is not None:
            graph.get_input_device().set_format(format_index)
        graph.add_sample_grabber(on_frame)
        graph.add_null_render()
        graph.prepare_preview_graph()
        graph.run()

        deadline = time.monotonic() + timeout
        while not arrived.is_set() and time.monotonic() < deadline:
            graph.grab_frame()
            time.sleep(1 / 30)

        if not arrived.is_set():
            raise TimeoutError("映像フレームが届きませんでした（入力信号を確認してください）。")
        return holder["frame"]
    finally:
        if graph is not None:
            try:
                graph.stop()
            except Exception:  # noqa: BLE001
                pass


def grab_single_frame(device_index: int, timeout: float = 8.0):
    """Capture one frame as an RGB uint8 array, or raise on failure.

    Used by colour calibration, which needs a still rather than a live feed.
    Safe to call from a worker thread: it initialises COM for that thread.

    Whatever format the device happens to be sitting in cannot always be
    bridged to RGB24 - OBS's virtual camera parks on I420 and the graph
    refuses to connect - so if the device's own setting fails, pin a format
    known to convert and try again.
    """
    import comtypes
    from pygrabber.dshow_graph import FilterGraph

    comtypes.CoInitialize()
    try:
        attempts: list[int | None] = [None]
        try:
            probe = FilterGraph()
            probe.add_video_input_device(device_index)
            available = _enumerate_formats(probe.get_input_device())
            ranked = sorted(
                (f for f in available if f.subtype in _PREFERRED_SUBTYPES),
                key=lambda f: (-f.pixels, _PREFERRED_SUBTYPES.index(f.subtype)),
            )
            attempts += [f.index for f in ranked[:5]]
        except Exception:  # noqa: BLE001 - fall back to the default attempt
            pass

        failures: list[str] = []
        for format_index in attempts:
            try:
                return _grab_once(device_index, format_index, timeout)
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001 - try the next format
                label = "既定" if format_index is None else f"format#{format_index}"
                failures.append(f"{label}: {type(exc).__name__}")

        # Every format failing to even build a graph almost always means the
        # device is already open elsewhere; a capture device admits one app.
        raise RuntimeError(
            "デバイスを開けませんでした。OBS など、このデバイスを使用中のアプリを"
            "終了してから再試行してください。\n"
            "（OBS を動かしたまま測りたい場合は、デバイス欄で OBS Virtual Camera を"
            "選んでください。）\n"
            "試したフォーマット: " + ", ".join(failures)
        )
    finally:
        comtypes.CoUninitialize()


def _locate(name: str) -> tuple[ut.Topology | None, str, str]:
    """Map a DirectShow display name onto its USB node and driver.

    A composite device exposes one child interface per function, so the same
    name matches both the camera and the audio interface. We want the camera
    one, otherwise we would report the audio driver as the video driver.
    """
    nodes = ut.find_instances_by_name(name)
    if not nodes:
        return None, "", ""

    services = {node.instance_id: ut.get_service(node.instance_id) for node in nodes}
    chosen = next(
        (n for n in nodes if services[n.instance_id].lower() == "usbvideo"),
        next((n for n in nodes if "&MI_00" in n.instance_id.upper()), nodes[0]),
    )
    return ut.trace(chosen.instance_id), services[chosen.instance_id], chosen.instance_id


def enumerate_devices(with_topology: bool = True) -> list[CaptureDevice]:
    try:
        names = FilterGraph().get_input_devices()
    except Exception as exc:  # noqa: BLE001 - no DirectShow at all
        raise RuntimeError(f"DirectShow を初期化できませんでした: {exc}") from exc

    devices: list[CaptureDevice] = []
    for i, name in enumerate(names):
        formats: list[VideoFormat] = []
        current = None
        error = ""
        try:
            formats, current = _read_formats(i)
        except Exception as exc:  # noqa: BLE001 - report per-device, keep going
            error = f"{type(exc).__name__}: {exc}"

        device = CaptureDevice(index=i, name=name, formats=formats, current=current, error=error)

        if with_topology and not device.is_virtual:
            device.topology, device.driver_service, device.instance_id = _locate(name)

        devices.append(device)
    return devices
