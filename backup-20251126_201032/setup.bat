@echo off
REM Setup script for NLP Text Classification Dashboard (Windows)

echo ========================================
echo NLP Text Classification Dashboard - Setup
echo ========================================
echo.

REM Check Python version
echo [1/6] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher from python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%
echo.

REM Create virtual environment
echo [2/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
    set /p RECREATE="Do you want to recreate it? (y/n): "
    if /i "%RECREATE%"=="y" (
        rmdir /s /q venv
        python -m venv venv
        echo Virtual environment recreated
    )
) else (
    python -m venv venv
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo pip upgraded
echo.

REM Install dependencies
echo [5/6] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
echo Dependencies installed
echo.

REM Download NLTK data
echo [6/6] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('brown', quiet=True); print('NLTK data downloaded')"
echo.

REM Create output directory
if not exist nlp_results mkdir nlp_results
echo Output directory created
echo.

REM Setup complete
echo ========================================
echo Setup complete!
echo.
echo To start the application:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run the app: streamlit run streamlit_nlp_app.py
echo.
echo For more information, see README.md
echo ========================================
echo.
pause
