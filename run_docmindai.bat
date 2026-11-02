@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo DocMind AI Python environment was not found.
  echo First, run the installation steps in README.md.
  pause
  exit /b 1
)
if not defined PORT set "PORT=8000"
echo Starting DocMind AI at http://127.0.0.1:%PORT%
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port %PORT%
