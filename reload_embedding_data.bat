@echo off
setlocal

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

set "LLM_PROVIDER=%~1"

echo Rechunk data locally...

if "%LLM_PROVIDER%"=="" (
  "%PYTHON_EXE%" "%~dp0rechunk_runtime.py"
) else (
  "%PYTHON_EXE%" "%~dp0rechunk_runtime.py" --provider "%LLM_PROVIDER%"
)

if errorlevel 1 (
  echo Rechunk failed.
  exit /b 1
)

echo Rechunk completed.
exit /b 0
