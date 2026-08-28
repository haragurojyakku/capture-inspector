"""Restart a device through PnP, which is the software equivalent of unplugging it.

`pnputil /restart-device` needs administrator rights, and this app has no reason
to run elevated the rest of the time, so the restart is launched as a separate
elevated process via the UAC prompt rather than by relaunching the whole GUI.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass

from . import usb_topology as ut

SEE_MASK_NOCLOSEPROCESS = 0x00000040
SEE_MASK_NO_CONSOLE = 0x00008000
SW_HIDE = 0

ERROR_CANCELLED = 1223
ERROR_SUCCESS_REBOOT_REQUIRED = 3010

INFINITE = 0xFFFFFFFF


class _ShellExecuteInfoW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("fMask", wintypes.ULONG),
        ("hwnd", wintypes.HWND),
        ("lpVerb", wintypes.LPCWSTR),
        ("lpFile", wintypes.LPCWSTR),
        ("lpParameters", wintypes.LPCWSTR),
        ("lpDirectory", wintypes.LPCWSTR),
        ("nShow", ctypes.c_int),
        ("hInstApp", wintypes.HINSTANCE),
        ("lpIDList", ctypes.c_void_p),
        ("lpClass", wintypes.LPCWSTR),
        ("hkeyClass", wintypes.HKEY),
        ("dwHotKey", wintypes.DWORD),
        ("hIcon", wintypes.HANDLE),
        ("hProcess", wintypes.HANDLE),
    ]


_shell32 = ctypes.WinDLL("shell32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(_ShellExecuteInfoW)]
_shell32.ShellExecuteExW.restype = wintypes.BOOL


def is_elevated() -> bool:
    try:
        return bool(_shell32.IsUserAnAdmin())
    except Exception:  # noqa: BLE001
        return False


@dataclass
class RestartResult:
    ok: bool
    message: str
    reboot_required: bool = False


def resolve_restart_target(instance_id: str) -> str:
    """Pick the node worth restarting: the composite device, not one interface.

    A capture card exposes separate camera/audio/HID interfaces as children of a
    single composite device. Restarting one child leaves the others untouched,
    so walk up to the composite - that is what re-enumerates the whole card.
    """
    topo = ut.trace(instance_id)
    for node in topo.chain:
        upper = node.instance_id.upper()
        if not upper.startswith("USB\\"):
            break
        if "&MI_" in upper:
            continue
        if "ROOT_HUB" in upper or "HUB" in node.label.upper():
            break
        return node.instance_id
    return instance_id


def restart_device(instance_id: str) -> RestartResult:
    """Disable and re-enable the device, prompting for elevation if needed."""
    if not instance_id:
        return RestartResult(False, "対象のデバイスを特定できませんでした。")

    pnputil = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "pnputil.exe")
    params = f'/restart-device "{instance_id}"'

    info = _ShellExecuteInfoW()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NO_CONSOLE
    info.hwnd = None
    info.lpVerb = None if is_elevated() else "runas"
    info.lpFile = pnputil
    info.lpParameters = params
    info.nShow = SW_HIDE

    if not _shell32.ShellExecuteExW(ctypes.byref(info)):
        err = ctypes.get_last_error()
        if err == ERROR_CANCELLED:
            return RestartResult(False, "管理者権限の昇格がキャンセルされました。")
        return RestartResult(False, f"実行を開始できませんでした (Win32 error {err})。")

    if not info.hProcess:
        return RestartResult(False, "プロセスハンドルを取得できませんでした。")

    try:
        _kernel32.WaitForSingleObject(info.hProcess, INFINITE)
        code = wintypes.DWORD()
        _kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
        rc = code.value
    finally:
        _kernel32.CloseHandle(info.hProcess)

    if rc == 0:
        return RestartResult(True, "デバイスを再初期化しました。")
    if rc == ERROR_SUCCESS_REBOOT_REQUIRED:
        return RestartResult(True, "再初期化しましたが、完全に反映するには再起動が必要です。", reboot_required=True)
    return RestartResult(False, f"pnputil がエラーを返しました (コード {rc})。")
