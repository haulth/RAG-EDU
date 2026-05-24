@echo off
setlocal

cd /d %~dp0

set "PRESET=%~1"

if not exist "venv\Scripts\python.exe" (
  echo [ERROR] Khong tim thay venv\Scripts\python.exe
  echo Hay chay setup_env.bat truoc.
  pause
  exit /b 1
)

echo ====================================================================
echo CHAY FAQ TEST CUNG LUONG GIAO DIEN
echo ====================================================================
if "%PRESET%"=="" (
  echo Preset: dung cau hinh mac dinh trong local_app_settings.json
  venv\Scripts\python.exe run_random_faq_system_test.py
) else (
  echo Preset override: %PRESET%
  venv\Scripts\python.exe run_random_faq_system_test.py --provider-preset %PRESET%
)

pause
