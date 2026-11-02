@echo off
setlocal
cd /d "%~dp0"
echo Checking Python...

set "PYTHON_CMD="
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_CMD=py -3.12"
  echo Found Python via py -3.12
) else (
  python --version >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=python"
    echo Found Python in PATH
  ) else (
    py --version >nul 2>&1
    if not errorlevel 1 (
      set "PYTHON_CMD=py"
      echo Found Python via py launcher
    )
  )
)

if not defined PYTHON_CMD (
  echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/
  pause
  exit /b 1
)

%PYTHON_CMD% --version
%PYTHON_CMD% -m venv .venv
if errorlevel 1 goto error
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto error
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
if errorlevel 1 goto error
echo.
echo Setup complete. Install Ollama and configure OLLAMA_URL and OLLAMA_MODEL
echo before starting the app. See the local setup section in README.md.
echo.
echo Start the website with run_docmindai.bat
pause
exit /b 0
:error
echo Setup failed. Read the error above, then check README.md.
pause
exit /b 1
