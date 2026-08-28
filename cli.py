"""Text report - the same analysis as the GUI, for when you just want to paste it somewhere."""
from __future__ import annotations

import sys

from capture_inspector.devices import enumerate_devices
from capture_inspector.diagnose import Severity, analyse, obs_recommendation

MARK = {
    Severity.CRITICAL: "[!!]",
    Severity.WARNING: "[! ]",
    Severity.OK: "[ok]",
    Severity.INFO: "[i ]",
}


def main() -> int:
    try:
        devices = enumerate_devices()
    except RuntimeError as exc:
        print(exc)
        return 1

    real = [d for d in devices if not d.is_virtual]
    if not real:
        print("キャプチャデバイスが見つかりませんでした。")
        return 1

    for device in real:
        print("=" * 72)
        print(f"  {device.name}")
        print("=" * 72)

        report = analyse(device)
        print(f"\n>>> {report.headline}\n")

        for finding in report.findings:
            print(f"{MARK[finding.severity]} {finding.title}")
            for line in finding.detail.splitlines():
                print(f"     {line}")
            if finding.fix:
                print("     --")
                for line in finding.fix.splitlines():
                    print(f"     -> {line}")
            print()

        if device.formats:
            print("--- 対応フォーマット " + "-" * 51)
            for res in device.resolutions:
                fps = device.fps_for(res)
                subs = sorted({f.subtype for f in device.formats if f.resolution == res})
                fps_str = ", ".join(f"{v:g}" for v in fps)
                print(f"  {res[0]:>5}x{res[1]:<5}  {fps_str:>12} fps   {', '.join(subs)}")
            print()
            print("--- 推奨設定 " + "-" * 58)
            print(obs_recommendation(device))
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
