@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Python virtual env not found at .venv\Scripts\python.exe
  echo Create it first and install requirements.
  exit /b 1
)

set "ARTIFACT_DIR=artifacts"
if not "%~1"=="" set "ARTIFACT_DIR=%~1"

if not exist "%ARTIFACT_DIR%" (
  echo [ERROR] Artifact directory not found: %ARTIFACT_DIR%
  echo Train first, for example:
  echo   .venv\Scripts\python -m src.main train --data-dir data --output-dir artifacts --device cpu
  exit /b 1
)

set "DEVICE=cuda"
if /i "%~2"=="cuda" set "DEVICE=cuda"

echo Starting QA console...
echo Artifact dir: %ARTIFACT_DIR%
echo Device: %DEVICE%
echo.

".venv\Scripts\python.exe" -m src.main qa --artifact-dir "%ARTIFACT_DIR%" --device %DEVICE% --top-k 3

endlocal
