"""Enumerate DirectShow capture devices and the video formats they advertise.

This is the same list OBS reads when it fills in its resolution/FPS dropdowns,
so if a resolution is missing here it will be missing in OBS too - which makes
this the right place to look when OBS "only offers the device default".
"""
from __future__ import annotations

from dataclasses import dataclass

from pygrabber.dshow_graph import FilterGraph

from . import usb_topology as ut


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


def _read_formats(graph_index: int) -> tuple[list[VideoFormat], tuple[int, int] | None]:
    graph = FilterGraph()
    graph.add_video_input_device(graph_index)
    device = graph.get_input_device()

    try:
        current = device.get_current_format()
    except Exception:  # noqa: BLE001 - some virtual cams refuse this
        current = None

    formats: list[VideoFormat] = []
    for raw in device.get_formats():
        # pygrabber derives 'min_framerate' from MinFrameInterval; a *smaller*
        # interval means a *higher* rate, so that field is the maximum fps.
        fps = max(raw["min_framerate"], raw["max_framerate"])
        formats.append(
            VideoFormat(
                index=raw["index"],
                width=raw["width"],
                height=abs(raw["height"]),
                fps=round(fps, 3),
                subtype=raw["media_type_str"],
            )
        )
    return formats, current


def grab_single_frame(device_index: int, timeout: float = 8.0):
    """Capture one frame as an RGB uint8 array, or raise on failure.

    Used by colour calibration, which needs a still rather than a live feed.
    Safe to call from a worker thread: it initialises COM for that thread.
    """
    import threading
    import time

    import comtypes
    from pygrabber.dshow_graph import FilterGraph

    comtypes.CoInitialize()
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
        graph.add_sample_grabber(on_frame)
        graph.add_null_render()
        graph.prepare_preview_graph()
        graph.run()

        deadline = time.monotonic() + timeout
        while not arrived.is_set() and time.monotonic() < deadline:
            graph.grab_frame()
            time.sleep(1 / 30)

        if not arrived.is_set():
            raise TimeoutError("映像フレームを取得できませんでした（入力信号を確認してください）。")
        return holder["frame"]
    finally:
        if graph is not None:
            try:
                graph.stop()
            except Exception:  # noqa: BLE001
                pass
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
