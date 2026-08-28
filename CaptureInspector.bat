@echo off
rem Launch the GUI using the project's own venv, from anywhere.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" "CaptureInspector.pyw"
