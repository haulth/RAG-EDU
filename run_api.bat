@echo off
setlocal

cd /d %~dp0

set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
set "API_SCRIPT=%~dp0api_server.py"
set "API_HEALTH_URL=http://127.0.0.1:8000/api/health"
set "APP_URL=http://127.0.0.1:8000/"

echo ====================================================================
echo KHOI DONG API + CHAT UI
echo ====================================================================

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Khong tim thay %PYTHON_EXE%
  echo Hay chay setup_env.bat truoc.
  pause
  exit /b 1
)

if not exist "%API_SCRIPT%" (
  echo [ERROR] Khong tim thay %API_SCRIPT%
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $null = Invoke-RestMethod -Uri '%API_HEALTH_URL%' -TimeoutSec 2; exit 0 } catch { exit 1 }"

if not errorlevel 1 (
  echo [INFO] Server da chay san hoac dang preload runtime.
  goto wait_for_ready
)

echo [1/3] Khoi dong server API...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process -FilePath '%PYTHON_EXE%' -ArgumentList '\"%API_SCRIPT%\"' -WorkingDirectory '%~dp0' -WindowStyle Normal"

echo [2/3] Doi server san sang...
:wait_for_ready
for /L %%i in (1,1,180) do (
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { $resp = Invoke-RestMethod -Uri '%API_HEALTH_URL%' -TimeoutSec 2; if ($resp.loaded -eq $true) { exit 0 } else { exit 1 } } catch { exit 1 }"
  if not errorlevel 1 goto open_browser
  >nul timeout /t 1 /nobreak
)

echo [ERROR] Khong the khoi dong server trong thoi gian cho phep.
echo Hay xem cua so "ChatBot API Server" de kiem tra log.
pause
exit /b 1

:open_browser
echo [3/3] Mo giao dien chat tai %APP_URL%
start "" "%APP_URL%"
exit /b 0
