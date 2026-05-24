@echo off
setlocal

cd /d %~dp0

echo ====================================================================
echo KHOI DONG CHATBOT
echo ====================================================================
echo [INFO] run.bat nay da duoc chuyen sang UI thong nhat qua API server.

call "%~dp0run_api.bat"
exit /b %errorlevel%
