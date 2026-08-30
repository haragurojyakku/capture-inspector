"""Turn raw device facts into plain-language findings about why capture is limited."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .devices import CaptureDevice
from .usb_topology import LinkSpeed
from .usb_topology import list_usb3_controllers as ut_list_usb3

# Effective throughput, not the marketing headline: USB 2.0's 480 Mbps bus gives
# roughly 320-400 Mbps to bulk/isochronous payload after protocol overhead.
USB2_EFFECTIVE_BPS = 360_000_000
USB3_EFFECTIVE_BPS = 4_000_000_000

BYTES_PER_PIXEL = {"YUY2": 2, "UYVY": 2, "NV12": 1.5, "I420": 1.5, "YV12": 1.5, "RGB24": 3, "ARGB32": 4}


class Severity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    OK = "ok"
    INFO = "info"


@dataclass
class Finding:
    severity: Severity
    title: str
    detail: str
    fix: str = ""


@dataclass
class Report:
    device: CaptureDevice
    findings: list[Finding] = field(default_factory=list)

    @property
    def worst(self) -> Severity:
        for level in (Severity.CRITICAL, Severity.WARNING, Severity.INFO, Severity.OK):
            if any(f.severity is level for f in self.findings):
                return level
        return Severity.OK

    @property
    def headline(self) -> str:
        worst = self.worst
        if worst is Severity.CRITICAL:
            return "接続に問題があります - 本来の性能が出ていません"
        if worst is Severity.WARNING:
            return "動作していますが、気になる点があります"
        return "正常です"


def required_bps(width: int, height: int, fps: float, subtype: str) -> float:
    """Raw bandwidth an uncompressed stream of this shape needs, in bits/sec."""
    bpp = BYTES_PER_PIXEL.get(subtype.upper(), 2)
    return width * height * bpp * fps * 8


def _fmt_bps(bps: float) -> str:
    if bps >= 1_000_000_000:
        return f"{bps / 1_000_000_000:.2f} Gbps"
    return f"{bps / 1_000_000:.0f} Mbps"


def analyse(device: CaptureDevice) -> Report:
    report = Report(device=device)
    add = report.findings.append

    if device.error:
        add(
            Finding(
                Severity.CRITICAL,
                "デバイスを開けませんでした",
                f"フォーマット一覧を取得できませんでした。\n{device.error}",
                "仮想カメラの場合は、送出側（OBS など）で開始されているか確認してください。\n"
                "物理デバイスの場合は、そのデバイスを掴んでいるアプリを全て終了してから"
                "再試行してください。キャプチャデバイスは同時に1つのアプリからしか開けません。",
            )
        )
        return report

    if not device.formats:
        add(
            Finding(
                Severity.CRITICAL,
                "対応フォーマットが1つも報告されません",
                "デバイスは見えていますが、映像フォーマットを何も返しません。",
                "HDMI入力元の電源が入っているか、ケーブルが挿さっているかを確認してください。",
            )
        )
        return report

    topo = device.topology
    speed = topo.link_speed if topo else LinkSpeed.UNKNOWN
    best = device.best

    # --- The headline check: is the card even on a USB 3 path? ---------------
    if speed is LinkSpeed.HIGHSPEED:
        controller = topo.host_controller.label if topo and topo.host_controller else "USB 2.0 コントローラー"
        detail = (
            f"このデバイスは USB 2.0 の経路にぶら下がっています。\n"
            f"接続先: {controller}\n\n"
            f"USB 2.0 の実効帯域は約 {_fmt_bps(USB2_EFFECTIVE_BPS)} しかありません。"
        )
        need = required_bps(1920, 1080, 60, "YUY2")
        detail += (
            f"\n1920x1080 @ 60fps (YUY2 非圧縮) には {_fmt_bps(need)} 必要で、"
            f"USB 2.0 の約 {need / USB2_EFFECTIVE_BPS:.1f} 倍です。"
            "\n\nそのためカード側が自分で低いフォーマットだけを申告しており、"
            "OBS にも申告された分しか出てきません。ドライバの問題ではありません。"
        )
        add(
            Finding(
                Severity.CRITICAL,
                "USB 2.0 で接続されています（これが原因です）",
                detail,
                "キャプチャカードを USB 3.0 以上のポート（青い端子 / SS マーク）に挿し替えてください。"
                "USBハブ経由なら、ハブ自体の上流ケーブルも USB 3.0 ポートに挿す必要があります。"
                "付属の USB 3.0 ケーブルを使うことも重要です（USB 2.0 ケーブルでは同じ症状になります）。",
            )
        )
        controllers = ut_list_usb3()
        if controllers:
            add(
                Finding(
                    Severity.INFO,
                    f"このPCには USB 3.0 コントローラーが {len(controllers)} 基あります",
                    "\n".join(f"  ・{c.label}" for c in controllers)
                    + "\n\nつまり挿し替え先は存在します。上記コントローラーにつながるポートを探してください。",
                    "デスクトップPCなら背面の青い USB 端子が候補です。"
                    "どのポートがどのコントローラーかは、実際に挿し替えてから本ツールで『再スキャン』すれば確認できます。",
                )
            )
    elif speed is LinkSpeed.SUPERSPEED:
        add(
            Finding(
                Severity.OK,
                "USB 3.x で接続されています",
                f"接続先: {topo.host_controller.label if topo and topo.host_controller else 'xHCI コントローラー'}",
            )
        )

    # --- Hubs in the path ----------------------------------------------------
    if topo and topo.hub_count > 0:
        add(
            Finding(
                Severity.WARNING if speed is LinkSpeed.SUPERSPEED else Severity.INFO,
                f"USBハブを {topo.hub_count} 段経由しています",
                "\n".join(f"{'  ' * i}└ {n.label}" for i, n in enumerate(topo.usb_chain[1:], start=1)),
                "可能ならPC本体のポートに直接挿してください。ハブ経由は帯域を他機器と分け合うため、"
                "高解像度キャプチャではコマ落ちの原因になります。",
            )
        )

    # --- What the card actually offers --------------------------------------
    if best is not None:
        if best.pixels < 1920 * 1080:
            add(
                Finding(
                    Severity.WARNING,
                    f"最大解像度が {best.width}x{best.height} 止まりです",
                    f"報告されたフォーマットは {len(device.formats)} 種類のみで、1920x1080 が含まれていません。\n"
                    f"最高: {best}",
                    "この製品は本来 1080p60 以上に対応します。上のUSB接続の項目を先に解決してください。",
                )
            )
        else:
            add(
                Finding(
                    Severity.OK,
                    f"最大 {best.width}x{best.height} @ {best.fps:g}fps を利用できます",
                    f"報告されたフォーマット: {len(device.formats)} 種類 / 形式: {', '.join(device.subtypes)}",
                )
            )

    # --- Driver expectations -------------------------------------------------
    if device.driver_service:
        if device.driver_service.lower() == "usbvideo":
            add(
                Finding(
                    Severity.INFO,
                    "標準UVCドライバ (usbvideo) で動作しています",
                    "これは異常ではありません。この種のキャプチャカードは UVC 準拠で、"
                    "Windows標準ドライバで動くのが正常な状態です。"
                    "メーカー製ソフトを入れても、この項目は変わりませんし、変える必要もありません。",
                )
            )
        else:
            add(
                Finding(
                    Severity.INFO,
                    f"ドライバ: {device.driver_service}",
                    "メーカー専用ドライバで動作しています。",
                )
            )

    return report


def practical_choice(device: CaptureDevice):
    """The format most people actually want, which is rarely the biggest one.

    Cards top out at 4K30 but game capture wants motion, so 1080p60 beats
    2160p30. Among equal candidates prefer NV12: it carries the same picture in
    3/4 the bandwidth of YUY2, which matters on a shared USB bus.
    """
    if not device.formats:
        return None

    def rank(fmt):
        return (fmt.subtype.upper() == "NV12", fmt.pixels)

    smooth = [f for f in device.formats if f.fps >= 59]
    if smooth:
        target = [f for f in smooth if f.width == 1920 and f.height == 1080]
        return max(target or smooth, key=rank)
    return max(device.formats, key=lambda f: (f.pixels, f.fps, f.subtype.upper() == "NV12"))


def obs_recommendation(device: CaptureDevice) -> str:
    """Concrete settings to type into OBS for this device, as it is right now."""
    pick = practical_choice(device)
    if pick is None:
        return "利用可能なフォーマットがないため、推奨設定を出せません。"

    lines = [
        "OBS の「映像キャプチャデバイス」プロパティでの設定:",
        "",
        f"  デバイス          : {device.name}",
        "  解像度/FPS タイプ : カスタム",
        f"  解像度            : {pick.width}x{pick.height}",
        f"  FPS               : {pick.fps:g}",
        f"  映像フォーマット  : {pick.subtype}",
        "",
    ]

    best = device.best
    if best is not None and best.resolution != pick.resolution:
        lines.append(
            f"※ 最大は {best.width}x{best.height} @ {best.fps:g}fps ですが、"
            f"ゲーム用途なら解像度より fps を優先した上の設定を推奨します。"
        )
        lines.append("")

    lines += [
        "補足: 「解像度/FPS タイプ」を『デバイスの既定値』のままにすると、",
        "カードが最初に申告した1つの設定に固定されます。『カスタム』に切り替えると",
        "下の一覧にある組み合わせを自由に選べます。",
    ]
    return "\n".join(lines)
