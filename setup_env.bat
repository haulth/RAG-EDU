@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Script tạo môi trường ảo và cài đặt dependencies cho ChatBot
REM ============================================================================

echo ====================================================================
echo TAO MOI TRUONG AO CHO CHATBOT
echo ====================================================================

REM --------------------------------------------------------------------
REM Kiem tra Python 3.9
REM --------------------------------------------------------------------
py -3.9 --version
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.9 chua duoc cai dat!
    echo Vui long cai Python 3.9.13
    echo https://www.python.org/downloads/release/python-3913/
    pause
    exit /b
)

echo [OK] Tim thay Python 3.9!

REM --------------------------------------------------------------------
echo.
echo [1/5] Tao moi truong ao voi Python 3.9...
py -3.9 -m venv venv

if errorlevel 1 (
    echo [ERROR] Tao venv that bai!
    pause
    exit /b
)

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Khong tim thay venv!
    pause
    exit /b
)

echo [OK] Da tao moi truong ao thanh cong!

REM --------------------------------------------------------------------
echo.
echo [2/5] Kich hoat moi truong ao...
call venv\Scripts\activate.bat

REM --------------------------------------------------------------------
echo.
echo [3/5] Nang cap pip...
venv\Scripts\python.exe -m pip install --upgrade pip

if errorlevel 1 (
    echo [ERROR] Loi khi update pip
    pause
)

REM --------------------------------------------------------------------
echo.
echo [4/5] Cai dat PyTorch (CUDA 11.8)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if errorlevel 1 (
    echo [ERROR] Loi khi cai PyTorch
    pause
)

REM --------------------------------------------------------------------
echo.
echo [5/5] Cai dat cac thu vien...

if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo Cai dat thu vien co ban...
    pip install fastapi uvicorn transformers sentence-transformers faiss-cpu rank-bm25
    pip install numpy pandas PyMuPDF huggingface-hub tqdm nltk matplotlib
    pip install accelerate bitsandbytes
)

if errorlevel 1 (
    echo [ERROR] Loi khi cai thu vien
    pause
)

REM --------------------------------------------------------------------
echo.
echo [5.1/5] Tao file local settings neu chua co...
if not exist "local_app_settings.json" (
    if exist "local_app_settings.example.json" (
        copy /Y "local_app_settings.example.json" "local_app_settings.json" >nul
        echo [OK] Da tao local_app_settings.json tu file mau.
    )
)

REM --------------------------------------------------------------------
echo.
echo ====================================================================
echo TAI MODELS TU HUGGING FACE
echo ====================================================================

if exist "download_models.py" (
    echo.
    set /p DOWNLOAD="Ban co muon tai models ngay bay gio (~12GB tuy thuoc vao toc do mang)? (y/n): "

    if /I "%DOWNLOAD%"=="y" (
        echo Dang tai models tu HuggingFace...
        python download_models.py

        if errorlevel 1 (
            echo [WARNING] Loi khi tai models!
            echo Ban co the tai lai sau bang lenh: python download_models.py
            pause
        ) else (
            echo [OK] Da tai models thanh cong!
        )
    ) else (
        echo Ban da chon tai sau.
        echo Khi can tai models, chay lenh:
        echo python download_models.py
        pause
    )
)

echo.
echo ====================================================================
echo HOAN TAT CAI DAT!
echo ====================================================================

pause
