"""Tkinter front-end: one window that answers 'why can't I pick 1080p60?'."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from .device_control import is_elevated, resolve_restart_target, restart_device
from .devices import CaptureDevice, enumerate_devices
from .diagnose import (
    USB2_EFFECTIVE_BPS,
    USB3_EFFECTIVE_BPS,
    Severity,
    analyse,
    obs_recommendation,
    practical_choice,
    required_bps,
)
from .usb_topology import LinkSpeed

BG = "#f5f6f8"
CARD = "#ffffff"
INK = "#1c2024"
MUTED = "#6b7280"
BORDER = "#dfe3e8"

ACCENT = {
    Severity.CRITICAL: "#d92d20",
    Severity.WARNING: "#dc7609",
    Severity.OK: "#0f8a4a",
    Severity.INFO: "#2563eb",
}
BANNER_BG = {
    Severity.CRITICAL: "#fdecea",
    Severity.WARNING: "#fdf3e6",
    Severity.OK: "#e9f7ef",
    Severity.INFO: "#eaf1fd",
}
LABEL = {
    Severity.CRITICAL: "重大",
    Severity.WARNING: "注意",
    Severity.OK: "正常",
    Severity.INFO: "情報",
}


def _fmt_bps(bps: float) -> str:
    if bps >= 1_000_000_000:
        return f"{bps / 1_000_000_000:.2f} Gbps"
    return f"{bps / 1_000_000:.0f} Mbps"


class ScrollFrame(ttk.Frame):
    """A vertically scrollable container that tracks the canvas width."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)

        bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.body = ttk.Frame(self.canvas, style="Body.TFrame")
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_wheel(self, event: tk.Event) -> None:
        if self.winfo_exists() and str(self.canvas.winfo_containing(event.x_root, event.y_root)).startswith(
            str(self.canvas)
        ):
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def clear(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()


class Preview:
    """Runs the capture graph on a worker thread and paints frames on the Tk thread.

    Two threads are unavoidable here: DirectShow delivers buffers on its own
    streaming thread, and Tk may only be touched from the thread that owns the
    widgets. So the callback does nothing but stash the newest frame, and a
    timer on the main thread picks it up. Anything drawn straight from the
    callback silently fails to appear.
    """

    TICK_MS = 50
    NO_SIGNAL_AFTER = 4.0

    def __init__(self, canvas: tk.Canvas, status: tk.StringVar) -> None:
        self.canvas = canvas
        self.status = status
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._messages: queue.Queue[tuple[bool, str]] = queue.Queue()
        self._photo = None
        self._started_at = 0.0
        self._frames = 0
        self._ticking = False
        self._failed = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, device_index: int) -> None:
        if self.running:
            return
        self._stop.clear()
        with self._lock:
            self._latest = None
        self._frames = 0
        self._failed = False
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, args=(device_index,), daemon=True)
        self._thread.start()
        if not self._ticking:
            self._ticking = True
            self.canvas.after(self.TICK_MS, self._tick)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------- worker thread
    def _run(self, device_index: int) -> None:
        import comtypes
        from pygrabber.dshow_graph import FilterGraph

        # comtypes only initializes COM on the thread that imports it, so a
        # worker thread must do its own CoInitialize or every CoCreateInstance
        # fails with "CoInitialize has not been called".
        comtypes.CoInitialize()
        graph = None
        try:
            graph = FilterGraph()
            graph.add_video_input_device(device_index)
            graph.add_sample_grabber(self._on_frame)
            graph.add_null_render()
            graph.prepare_preview_graph()
            graph.run()
            self._messages.put((False, "プレビュー中"))
            while not self._stop.wait(1 / 30):
                graph.grab_frame()
        except Exception as exc:  # noqa: BLE001 - device busy is the common case
            self._messages.put((
                True,
                f"プレビューを開始できません: {type(exc).__name__}: {exc} / "
                "OBS など他のアプリがこのデバイスを使用中の可能性があります。",
            ))
        finally:
            if graph is not None:
                try:
                    graph.stop()
                except Exception:  # noqa: BLE001
                    pass
            self._messages.put((False, "停止中"))
            comtypes.CoUninitialize()

    def _on_frame(self, image) -> None:
        """Called on the DirectShow streaming thread - no Tk calls allowed here.

        pygrabber has already flipped the rows upright; the remaining fix is
        that DirectShow's RGB24 is BGR in memory.
        """
        try:
            frame = image[:, :, ::-1].copy()
        except Exception:  # noqa: BLE001 - never let a bad buffer kill the stream
            return
        with self._lock:
            self._latest = frame

    # --------------------------------------------------------- main thread
    def _tick(self) -> None:
        # Drain everything queued this tick, but never let a routine "stopped"
        # notice bury the error that caused the stop.
        while True:
            try:
                is_error, text = self._messages.get_nowait()
            except queue.Empty:
                break
            if is_error:
                self._failed = True
                self.status.set(text)
            elif not self._failed:
                self.status.set(text)

        with self._lock:
            frame = self._latest
            self._latest = None

        if frame is not None:
            self._frames += 1
            self._draw(frame)
        elif (
            self.running
            and not self._failed
            and self._frames == 0
            and time.monotonic() - self._started_at > self.NO_SIGNAL_AFTER
        ):
            self.status.set("映像信号が来ていません（HDMI入力元の電源・ケーブルを確認してください）")

        if self.running or not self._messages.empty():
            self.canvas.after(self.TICK_MS, self._tick)
        else:
            self._ticking = False

    def _draw(self, frame) -> None:
        from PIL import Image, ImageTk

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 10 or height < 10:
            return
        try:
            pil = Image.fromarray(frame)
            source = f"{pil.width}x{pil.height}"
            pil.thumbnail((width, height), Image.BILINEAR)
            photo = ImageTk.PhotoImage(pil)
            self.canvas.delete("all")
            self.canvas.create_image(width // 2, height // 2, image=photo)
            self._photo = photo  # keep a reference or Tk drops the image
            self.status.set(f"プレビュー中  {source}  ({self._frames} フレーム受信)")
        except Exception as exc:  # noqa: BLE001
            self.status.set(f"描画エラー: {exc}")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CaptureInspector - キャプチャデバイス診断")
        self.geometry("1000x760")
        self.minsize(860, 600)
        self.configure(bg=BG)

        self.devices: list[CaptureDevice] = []
        self.device_var = tk.StringVar()
        self.preview_status = tk.StringVar(value="停止中")
        self.color_status = tk.StringVar(value="未測定")
        self.color_report = None

        self._init_style()
        self._build()
        self.after(100, self.scan)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- styling
    def _init_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Body.TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=INK, font=("Yu Gothic UI", 10))
        style.configure("Card.TLabel", background=CARD, foreground=INK, font=("Yu Gothic UI", 10))
        style.configure("CardTitle.TLabel", background=CARD, foreground=INK, font=("Yu Gothic UI", 11, "bold"))
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Yu Gothic UI", 9))
        style.configure("H1.TLabel", background=BG, foreground=INK, font=("Yu Gothic UI", 15, "bold"))
        style.configure("TButton", font=("Yu Gothic UI", 10), padding=(12, 6))
        style.configure("Treeview", font=("Consolas", 10), rowheight=24, background=CARD, fieldbackground=CARD)
        style.configure("Treeview.Heading", font=("Yu Gothic UI", 10, "bold"))
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Yu Gothic UI", 10), padding=(16, 8))

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        top = ttk.Frame(self, padding=(16, 14, 16, 8))
        top.pack(fill="x")

        ttk.Label(top, text="デバイス", style="TLabel").pack(side="left")
        self.combo = ttk.Combobox(top, textvariable=self.device_var, state="readonly", width=44,
                                  font=("Yu Gothic UI", 10))
        self.combo.pack(side="left", padx=(10, 10))
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self.render())

        ttk.Button(top, text="再スキャン", command=self.scan).pack(side="left")
        self.btn_restart = ttk.Button(top, text="デバイスを再初期化", command=self.restart_device)
        self.btn_restart.pack(side="left", padx=(8, 0))
        ttk.Button(top, text="レポートをコピー", command=self.copy_report).pack(side="right")

        self.banner = tk.Frame(self, bg=BANNER_BG[Severity.INFO], height=64)
        self.banner.pack(fill="x", padx=16, pady=(6, 10))
        self.banner.pack_propagate(False)
        self.banner_text = tk.Label(self.banner, text="スキャン中...", bg=BANNER_BG[Severity.INFO],
                                    fg=ACCENT[Severity.INFO], font=("Yu Gothic UI", 13, "bold"), anchor="w")
        self.banner_text.pack(fill="both", expand=True, padx=18)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self.tab_diag = ScrollFrame(self.nb)
        self.tab_formats = ttk.Frame(self.nb)
        self.tab_usb = ttk.Frame(self.nb)
        self.tab_preview = ttk.Frame(self.nb)
        self.tab_color = ttk.Frame(self.nb)

        self.nb.add(self.tab_diag, text="  診断  ")
        self.nb.add(self.tab_formats, text="  対応フォーマット  ")
        self.nb.add(self.tab_usb, text="  USB接続  ")
        self.nb.add(self.tab_preview, text="  プレビュー  ")
        self.nb.add(self.tab_color, text="  色校正  ")

        self._build_formats_tab()
        self._build_usb_tab()
        self._build_preview_tab()
        self._build_color_tab()

    def _build_formats_tab(self) -> None:
        wrap = ttk.Frame(self.tab_formats, padding=12)
        wrap.pack(fill="both", expand=True)

        cols = ("res", "fps", "sub", "aspect", "bw")
        self.tree = ttk.Treeview(wrap, columns=cols, show="headings", selectmode="browse")
        for col, text, width, anchor in (
            ("res", "解像度", 130, "center"),
            ("fps", "FPS", 80, "center"),
            ("sub", "形式", 100, "center"),
            ("aspect", "アスペクト比", 110, "center"),
            ("bw", "必要帯域(非圧縮)", 160, "e"),
        ):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor)

        bar = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("over2", foreground="#b42318")

        self.format_note = ttk.Label(self.tab_formats, text="", style="TLabel", wraplength=940, justify="left")
        self.format_note.pack(fill="x", padx=14, pady=(0, 12))

    def _build_usb_tab(self) -> None:
        wrap = ttk.Frame(self.tab_usb, padding=12)
        wrap.pack(fill="both", expand=True)
        self.usb_text = tk.Text(wrap, wrap="word", font=("Consolas", 10), bg=CARD, fg=INK,
                                relief="flat", padx=14, pady=12, highlightthickness=1,
                                highlightbackground=BORDER)
        self.usb_text.pack(fill="both", expand=True)
        self.usb_text.configure(state="disabled")

    def _build_preview_tab(self) -> None:
        wrap = ttk.Frame(self.tab_preview, padding=12)
        wrap.pack(fill="both", expand=True)

        bar = ttk.Frame(wrap)
        bar.pack(fill="x", pady=(0, 10))
        self.btn_preview = ttk.Button(bar, text="プレビュー開始", command=self.toggle_preview)
        self.btn_preview.pack(side="left")
        ttk.Label(bar, textvariable=self.preview_status, style="TLabel",
                  wraplength=700, justify="left").pack(side="left", padx=14)

        self.canvas = tk.Canvas(wrap, bg="#101214", highlightthickness=1, highlightbackground=BORDER)
        self.canvas.pack(fill="both", expand=True)
        self.preview = Preview(self.canvas, self.preview_status)

    def _build_color_tab(self) -> None:
        wrap = ttk.Frame(self.tab_color, padding=12)
        wrap.pack(fill="both", expand=True)

        steps = tk.Frame(wrap, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        steps.pack(fill="x", pady=(0, 10))
        tk.Label(
            steps,
            text="① テストパターンを保存 → ② スマホに転送し全画面表示 → ③ 測定 → ④ 補正LUTを保存",
            bg=CARD, fg=INK, font=("Yu Gothic UI", 10, "bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(
            steps,
            text="スマホの明るさ自動調整・Night Shift・ダークモードの色調整は切ってください。"
                 "測定するのは「送った色」と「返ってきた色」の差です。",
            bg=CARD, fg=MUTED, font=("Yu Gothic UI", 9), anchor="w", justify="left", wraplength=900,
        ).pack(fill="x", padx=14, pady=(0, 10))

        bar = ttk.Frame(wrap)
        bar.pack(fill="x", pady=(0, 10))
        ttk.Button(bar, text="① パターンを保存", command=self.save_pattern).pack(side="left")
        ttk.Button(bar, text="③ 測定する", command=self.run_calibration).pack(side="left", padx=(8, 0))
        self.btn_lut = ttk.Button(bar, text="④ 補正LUTを保存", command=self.save_lut, state="disabled")
        self.btn_lut.pack(side="left", padx=(8, 0))
        ttk.Label(bar, textvariable=self.color_status, style="TLabel",
                  wraplength=520, justify="left").pack(side="left", padx=14)

        self.color_out = tk.Text(wrap, wrap="word", font=("Consolas", 10), bg=CARD, fg=INK,
                                 relief="flat", padx=14, pady=12, highlightthickness=1,
                                 highlightbackground=BORDER)
        cbar = ttk.Scrollbar(wrap, orient="vertical", command=self.color_out.yview)
        self.color_out.configure(yscrollcommand=cbar.set, state="disabled")
        cbar.pack(side="right", fill="y")
        self.color_out.pack(fill="both", expand=True)

    # ---------------------------------------------------------------- colour
    def save_pattern(self) -> None:
        from tkinter import filedialog

        from .calibration import render_pattern

        device = self.selected
        # The chart is square so it survives either phone orientation; size it
        # from the capture's shorter side, which is what limits detail.
        side = 1080
        if device:
            pick = practical_choice(device)
            if pick:
                side = min(pick.width, pick.height)

        path = filedialog.asksaveasfilename(
            title="テストパターンの保存先",
            defaultextension=".png",
            initialfile=f"capture_test_pattern_{side}x{side}.png",
            filetypes=[("PNG image", "*.png")],
        )
        if not path:
            return
        render_pattern(side).save(path)
        self.color_status.set(
            f"保存しました ({side}x{side})。スマホでできるだけ大きく表示してください。"
        )

    def run_calibration(self) -> None:
        device = self.selected
        if device is None:
            return
        self.preview.stop()
        self.btn_preview.configure(text="プレビュー開始")
        self.color_status.set("測定中... フレームを取得しています")
        self.config(cursor="watch")

        def worker() -> None:
            from .calibration import analyse_colors, locate, measure
            from .devices import grab_single_frame

            try:
                frame = grab_single_frame(device.index)
                loc = locate(frame)
                if loc is None:
                    raise ValueError(
                        "テストパターンを検出できませんでした。\n"
                        "四隅のマーカー（赤・緑・青・黄）が4つとも画面に写っているか確認してください。"
                    )
                report = analyse_colors(measure(frame, loc), loc, frame.shape)
                self.after(0, lambda: self._show_calibration(report, loc, frame.shape))
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                self.after(0, lambda: self._calibration_failed(message))

        threading.Thread(target=worker, daemon=True).start()

    def _calibration_failed(self, message: str) -> None:
        self.config(cursor="")
        self.color_status.set("測定に失敗しました")
        self._write_color(message)

    def _show_calibration(self, report, loc, shape) -> None:
        from .calibration import LAYOUT

        self.config(cursor="")
        self.color_report = report
        self.btn_lut.configure(state="normal")
        self.color_status.set("測定完了")

        how = "四隅マーカー" if loc.method == "markers" else "外周フレーム（マーカー未検出）"
        chart_w, chart_h = loc.chart_size()
        box_w, box_h = loc.sample_box()
        coverage = (chart_w * chart_h) / (shape[1] * shape[0]) * 100
        lines = [
            f"フレーム {shape[1]}x{shape[0]} / 位置合わせ: {how}",
            f"チャート {chart_w:.0f}x{chart_h:.0f} px (フレームの {coverage:.1f}%)"
            f" / 1パッチの実測領域 {box_w:.0f}x{box_h:.0f} px",
        ]
        if loc.markers:
            lines.append("  " + "  ".join(
                f"{k.upper()}({v[0]:.0f},{v[1]:.0f})" for k, v in loc.markers.items()
            ))
        lines += [
            "",
            f"黒レベル : {report.black_level:7.2f}   (理想 0)",
            f"白レベル : {report.white_level:7.2f}   (理想 255)",
            f"ガンマ   : {report.gamma:7.4f}   (理想 1.0)",
            f"ゲイン   : R {report.gain[0]:.4f} / G {report.gain[1]:.4f} / B {report.gain[2]:.4f}",
            "",
            "--- 所見 " + "-" * 60,
        ]
        for finding in report.findings:
            lines.append(f"[{LABEL[finding.severity]}] {finding.title}")
            for line in finding.detail.splitlines():
                lines.append(f"    {line}")
            if finding.fix:
                for line in finding.fix.splitlines():
                    lines.append(f"  -> {line}")
            lines.append("")

        lines.append("--- 測定値 (送出 → 実測) " + "-" * 45)
        for patch in LAYOUT:
            m = report.measurements.get(patch.name)
            if m is None:
                continue
            delta = m - np.array(patch.ref, dtype=float)
            lines.append(
                f"  {patch.name:<12} 送出 {str(patch.ref):>16}  "
                f"実測 ({m[0]:6.1f},{m[1]:6.1f},{m[2]:6.1f})  "
                f"差 ({delta[0]:+6.1f},{delta[1]:+6.1f},{delta[2]:+6.1f})"
            )

        self._write_color("\n".join(lines))

    def _write_color(self, text: str) -> None:
        self.color_out.configure(state="normal")
        self.color_out.delete("1.0", "end")
        self.color_out.insert("1.0", text)
        self.color_out.configure(state="disabled")

    def save_lut(self) -> None:
        from tkinter import filedialog

        from .calibration import (
            MATRIX_SIGNIFICANT,
            build_1d_lut,
            build_3d_lut,
            fit_matrix,
            matrix_strength,
            tone_curves,
        )

        if self.color_report is None:
            return
        path = filedialog.asksaveasfilename(
            title="補正LUTの保存先",
            defaultextension=".cube",
            initialfile="capture_correction.cube",
            filetypes=[("Cube LUT", "*.cube")],
        )
        if not path:
            return

        try:
            strength = matrix_strength(fit_matrix(self.color_report, tone_curves(self.color_report, 64)))
            # A 1D LUT is exact for level and gamma errors and carries no grid
            # interpolation error, so only step up to 3D when there is a hue
            # error for it to fix - a 1D LUT cannot touch hue at all.
            use_3d = strength >= MATRIX_SIGNIFICANT
            text = build_3d_lut(self.color_report) if use_3d else build_1d_lut(self.color_report)
        except ValueError as exc:
            messagebox.showerror("LUTを生成できません", str(exc))
            return

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

        kind = (
            f"3D LUT を書き出しました（マトリクス誤差 {strength:.4f}）。\n"
            "色相のずれがあるため、トーンカーブに加えて 3x3 マトリクス補正を含めています。"
            if use_3d else
            f"1D LUT を書き出しました（マトリクス誤差 {strength:.4f} は軽微）。\n"
            "レベルとガンマのみの補正で、色相は変更しません。"
        )
        messagebox.showinfo(
            "保存しました",
            f"{path}\n\n{kind}\n\n"
            "OBS での適用方法:\n"
            "  1. 映像キャプチャデバイスのソースを右クリック → フィルタ\n"
            "  2. エフェクトフィルタに「LUT を適用」を追加\n"
            "  3. このファイルを指定\n\n"
            "注意: レンジ設定（色範囲）で直る場合は、LUT より先にそちらを直してください。\n"
            "クリップ（潰れ・飛び）は情報が失われているため復元できません。",
        )

    # ---------------------------------------------------------------- actions
    def scan(self) -> None:
        self.preview.stop()
        try:
            self.devices = enumerate_devices()
        except RuntimeError as exc:
            messagebox.showerror("エラー", str(exc))
            return

        names = [d.name for d in self.devices]
        self.combo["values"] = names
        if names:
            real = next((d.name for d in self.devices if not d.is_virtual), names[0])
            self.device_var.set(real)
        self.render()

    @property
    def selected(self) -> CaptureDevice | None:
        return next((d for d in self.devices if d.name == self.device_var.get()), None)

    def render(self) -> None:
        device = self.selected
        if device is None:
            self.banner_text.configure(text="キャプチャデバイスが見つかりません")
            return

        report = analyse(device)
        sev = report.worst
        self.banner.configure(bg=BANNER_BG[sev])
        self.banner_text.configure(text=report.headline, bg=BANNER_BG[sev], fg=ACCENT[sev])

        self._render_findings(report)
        self._render_formats(device)
        self._render_usb(device)

    def _render_findings(self, report) -> None:
        self.tab_diag.clear()
        body = self.tab_diag.body

        for finding in report.findings:
            accent = ACCENT[finding.severity]
            row = tk.Frame(body, bg=BG)
            row.pack(fill="x", pady=(0, 10), padx=2)

            tk.Frame(row, bg=accent, width=4).pack(side="left", fill="y")

            card = tk.Frame(row, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            card.pack(side="left", fill="both", expand=True)

            head = tk.Frame(card, bg=CARD)
            head.pack(fill="x", padx=14, pady=(11, 2))
            tk.Label(head, text=LABEL[finding.severity], bg=accent, fg="white",
                     font=("Yu Gothic UI", 8, "bold"), padx=7, pady=1).pack(side="left")
            tk.Label(head, text=finding.title, bg=CARD, fg=INK,
                     font=("Yu Gothic UI", 11, "bold"), anchor="w").pack(side="left", padx=8)

            tk.Label(card, text=finding.detail, bg=CARD, fg=INK, font=("Yu Gothic UI", 10),
                     justify="left", anchor="w", wraplength=860).pack(fill="x", padx=14, pady=(2, 8))

            if finding.fix:
                fix = tk.Frame(card, bg="#f4f6f9")
                fix.pack(fill="x", padx=14, pady=(0, 12))
                tk.Label(fix, text="対処", bg="#f4f6f9", fg=MUTED,
                         font=("Yu Gothic UI", 9, "bold"), anchor="w").pack(fill="x", padx=10, pady=(7, 0))
                tk.Label(fix, text=finding.fix, bg="#f4f6f9", fg=INK, font=("Yu Gothic UI", 10),
                         justify="left", anchor="w", wraplength=830).pack(fill="x", padx=10, pady=(1, 8))

        device = report.device
        if device.formats:
            card = tk.Frame(body, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            card.pack(fill="x", pady=(4, 12), padx=2)
            tk.Label(card, text="OBS 推奨設定", bg=CARD, fg=INK,
                     font=("Yu Gothic UI", 11, "bold"), anchor="w").pack(fill="x", padx=14, pady=(11, 4))
            tk.Label(card, text=obs_recommendation(device), bg=CARD, fg=INK,
                     font=("Consolas", 10), justify="left", anchor="w").pack(fill="x", padx=14, pady=(0, 12))

    def _render_formats(self, device: CaptureDevice) -> None:
        self.tree.delete(*self.tree.get_children())
        for fmt in sorted(device.formats, key=lambda f: (-f.pixels, -f.fps, f.subtype)):
            bps = required_bps(fmt.width, fmt.height, fmt.fps, fmt.subtype)
            gcd = _gcd(fmt.width, fmt.height) or 1
            tag = "over2" if bps > USB2_EFFECTIVE_BPS else ""
            self.tree.insert(
                "", "end",
                values=(f"{fmt.width}x{fmt.height}", f"{fmt.fps:g}", fmt.subtype,
                        f"{fmt.width // gcd}:{fmt.height // gcd}", _fmt_bps(bps)),
                tags=(tag,),
            )

        speed = device.topology.link_speed if device.topology else LinkSpeed.UNKNOWN
        budget = USB2_EFFECTIVE_BPS if speed is LinkSpeed.HIGHSPEED else USB3_EFFECTIVE_BPS
        self.format_note.configure(
            text=f"{len(device.formats)} 種類のフォーマットをデバイスが申告しています。"
                 f"OBS で選べるのはこの一覧が全てです。\n"
                 f"現在の接続 ({speed.value}) の実効帯域は約 {_fmt_bps(budget)}。"
                 f"赤字の行は USB 2.0 の帯域を超える組み合わせです。"
        )

    def _render_usb(self, device: CaptureDevice) -> None:
        topo = device.topology
        lines: list[str] = []
        if topo is None or not topo.chain:
            lines.append("USB デバイスとして特定できませんでした（仮想カメラ等の可能性があります）。")
        else:
            lines.append(f"リンク速度   : {topo.link_speed.value}")
            lines.append(f"ホスト制御   : {topo.host_controller.label if topo.host_controller else '不明'}")
            lines.append(f"経由ハブ数   : {topo.hub_count}")
            lines.append(f"ドライバ     : {device.driver_service or '不明'}")
            lines.append("")
            lines.append("接続経路 (デバイス → ホストコントローラー):")
            lines.append("")
            for i, node in enumerate(topo.usb_chain):
                lines.append(f"{'   ' * i}└─ {node.label}")
                lines.append(f"{'   ' * i}     {node.instance_id}")
            lines.append("")
            if topo.link_speed is LinkSpeed.HIGHSPEED:
                lines.append("※ 経路上に USB 2.0 のコントローラー／ルートハブがあります。")
                lines.append("  このコントローラー配下のポートである限り、どこに挿し替えても 480Mbps を超えられません。")
                lines.append("  別系統の USB 3.0 (xHCI) コントローラーのポートに挿し替える必要があります。")

        self.usb_text.configure(state="normal")
        self.usb_text.delete("1.0", "end")
        self.usb_text.insert("1.0", "\n".join(lines))
        self.usb_text.configure(state="disabled")

    def toggle_preview(self) -> None:
        device = self.selected
        if device is None:
            return
        if self.preview.running:
            self.preview.stop()
            self.btn_preview.configure(text="プレビュー開始")
            return
        self.preview_status.set("開始中...")
        self.preview.start(device.index)
        self.btn_preview.configure(text="プレビュー停止")

    def restart_device(self) -> None:
        """Disable/re-enable the card - the software version of replugging it."""
        device = self.selected
        if device is None or not device.instance_id:
            messagebox.showwarning(
                "再初期化できません",
                "USBデバイスとして特定できていないため、再初期化できません。\n"
                "仮想カメラなどは対象外です。",
            )
            return

        target = resolve_restart_target(device.instance_id)
        elevated_note = (
            "" if is_elevated() else "\n\n実行時に管理者権限の確認（UAC）が表示されます。"
        )
        proceed = messagebox.askokcancel(
            "デバイスを再初期化しますか？",
            f"次のデバイスを一度無効化してから有効化します。\n"
            f"USBケーブルを抜き差しするのと同じ効果です。\n\n"
            f"  {device.name}\n"
            f"  {target}\n\n"
            f"このデバイスを使用中のアプリ（OBS・録画ソフトなど）は\n"
            f"映像が途切れます。録画・配信中は実行しないでください。"
            f"{elevated_note}",
            icon=messagebox.WARNING,
            default=messagebox.CANCEL,
        )
        if not proceed:
            return

        self.preview.stop()
        self.btn_preview.configure(text="プレビュー開始")
        self.btn_restart.configure(state="disabled", text="再初期化中...")
        self.config(cursor="watch")

        def worker() -> None:
            result = restart_device(target)
            self.after(0, lambda: self._after_restart(result))

        threading.Thread(target=worker, daemon=True).start()

    def _after_restart(self, result) -> None:
        self.btn_restart.configure(state="normal", text="デバイスを再初期化")
        self.config(cursor="")

        if not result.ok:
            messagebox.showerror("再初期化に失敗しました", result.message)
            return

        # Re-enumeration is not instant; give the bus a moment before rescanning.
        self.after(2500, self.scan)
        messagebox.showinfo(
            "完了",
            f"{result.message}\n\n数秒後に自動で再スキャンします。"
            + ("\n\n再起動が必要と報告されました。" if result.reboot_required else ""),
        )

    def copy_report(self) -> None:
        device = self.selected
        if device is None:
            return
        report = analyse(device)
        out = [f"=== {device.name} ===", report.headline, ""]
        for f in report.findings:
            out.append(f"[{LABEL[f.severity]}] {f.title}")
            out.append(f.detail)
            if f.fix:
                out.append(f"対処: {f.fix}")
            out.append("")
        out.append("--- 対応フォーマット ---")
        for res in device.resolutions:
            fps = ", ".join(f"{v:g}" for v in device.fps_for(res))
            out.append(f"  {res[0]}x{res[1]}  {fps} fps")
        out.append("")
        out.append(obs_recommendation(device))

        self.clipboard_clear()
        self.clipboard_append("\n".join(out))
        messagebox.showinfo("コピーしました", "診断レポートをクリップボードにコピーしました。")

    def _on_close(self) -> None:
        self.preview.stop()
        self.destroy()


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def main() -> None:
    App().mainloop()
