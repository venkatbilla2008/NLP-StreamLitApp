@echo off
REM update_github_repo.bat
REM Script to update your GitHub repository with only essential files
REM For Windows systems

SETLOCAL ENABLEDELAYEDEXPANSION

echo ==========================================
echo Update GitHub Repository
echo ==========================================
echo.

REM Configuration
set REPO_URL=https://github.com/venkatbilla2008/nlp-text-pipeline.git
set REPO_DIR=nlp-text-pipeline
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set BACKUP_DIR=..\backup-%dt:~0,8%_%dt:~8,6%

echo Repository: %REPO_URL%
echo Local directory: %REPO_DIR%
echo.

REM Step 1: Clone or navigate to repository
if exist "%REPO_DIR%" (
    echo [OK] Repository directory exists
    cd "%REPO_DIR%"
) else (
    echo [*] Cloning repository...
    git clone "%REPO_URL%"
    if errorlevel 1 (
        echo [ERROR] Failed to clone repository
        echo Please check your internet connection and git installation
        pause
        exit /b 1
    )
    cd "%REPO_DIR%"
)

echo [OK] In repository directory: %CD%
echo.

REM Step 2: Backup existing files
echo [*] Creating backup of current files...
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"
xcopy * "%BACKUP_DIR%\" /E /I /Q /Y >nul 2>&1
echo [OK] Backup created at: %BACKUP_DIR%
echo.

REM Step 3: Remove all files (keep .git)
echo [*] Removing old files...
for /f "delims=" %%i in ('dir /b /a-d') do (
    if not "%%i"==".git" (
        del /f /q "%%i" >nul 2>&1
    )
)
for /f "delims=" %%i in ('dir /b /ad') do (
    if not "%%i"==".git" (
        rmdir /s /q "%%i" >nul 2>&1
    )
)
echo [OK] Old files removed
echo.

REM Step 4: Check for required files
echo [*] Checking for required files...
echo.
echo Please ensure you have these files ready:
echo   1. streamlit_nlp_app.py (your modified code)
echo   2. requirements.txt (with googletrans)
echo   3. README.md
echo.

set /p READY="Do you have all files ready? (y/n): "
if /i not "%READY%"=="y" (
    echo [ERROR] Please prepare the files first, then run this script again
    pause
    exit /b 1
)

echo.
echo [*] Please copy your files now:
echo.
echo Copy to: %CD%
echo.
echo Required files:
echo   * streamlit_nlp_app.py
echo   * requirements.txt
echo   * README.md
echo.

pause
echo.

REM Step 5: Create .gitignore
echo [*] Creating .gitignore...
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *.pyc
echo .Python
echo *.so
echo.
echo # Virtual Environment
echo venv/
echo env/
echo ENV/
echo .venv
echo.
echo # Streamlit
echo .streamlit/secrets.toml
echo.
echo # Output files
echo nlp_results/
echo *.parquet
echo *.csv
echo *.xlsx
echo.
echo # IDE
echo .vscode/
echo .idea/
echo *.swp
echo .DS_Store
echo.
echo # Logs
echo *.log
echo.
echo # OS
echo Thumbs.db
) > .gitignore
echo [OK] .gitignore created
echo.

REM Step 6: Create Streamlit config (optional)
echo [*] Creating .streamlit\config.toml...
if not exist ".streamlit" mkdir ".streamlit"
(
echo [theme]
echo primaryColor = "#1f77b4"
echo backgroundColor = "#ffffff"
echo secondaryBackgroundColor = "#f0f2f6"
echo textColor = "#262730"
echo.
echo [server]
echo maxUploadSize = 200
echo enableXsrfProtection = true
echo.
echo [browser]
echo gatherUsageStats = false
) > .streamlit\config.toml
echo [OK] Streamlit config created
echo.

REM Step 7: Verify files
echo [*] Verifying files...
echo.

if not exist "streamlit_nlp_app.py" (
    echo [ERROR] streamlit_nlp_app.py not found!
    echo.
    echo Please copy streamlit_nlp_app.py to this directory:
    echo %CD%
    echo.
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    echo.
    echo Please copy requirements.txt to this directory:
    echo %CD%
    echo.
    pause
    exit /b 1
)

if not exist "README.md" (
    echo [WARNING] README.md not found (recommended but not required)
)

echo [OK] Essential files verified
echo.

REM Step 8: Check git status
echo [*] Current files:
dir /b
echo.

REM Step 9: Git add, commit, push
echo [*] Git operations...
echo.

REM Add all files
git add .

REM Show status
echo Git status:
git status --short
echo.

REM Commit
set /p COMMIT_MSG="Enter commit message (or press Enter for default): "
if "%COMMIT_MSG%"=="" (
    set COMMIT_MSG=Update: NLP app with translation support
)

git commit -m "%COMMIT_MSG%"
if errorlevel 1 (
    echo [WARNING] Commit failed or no changes to commit
    echo.
)
echo [OK] Changes committed
echo.

REM Push
echo [*] Pushing to GitHub...
echo.
echo You may be prompted for GitHub credentials:
echo   Username: Your GitHub username
echo   Password: Your Personal Access Token (NOT your password!)
echo.
echo If you don't have a token:
echo   1. Go to: https://github.com/settings/tokens
echo   2. Generate new token (classic)
echo   3. Check 'repo' scope
echo   4. Copy the token
echo   5. Use it as password when prompted
echo.

git push origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Push failed!
    echo.
    echo Common fixes:
    echo.
    echo 1. If remote has changes:
    echo    git pull origin main
    echo    git push origin main
    echo.
    echo 2. If you want to force overwrite (CAREFUL!):
    echo    git push origin main --force
    echo.
    echo 3. If authentication failed:
    echo    - Use Personal Access Token as password
    echo    - Get it from: https://github.com/settings/tokens
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ==========================================
    echo SUCCESS!
    echo ==========================================
    echo.
    echo Your repository has been updated!
    echo.
    echo [+] View on GitHub:
    echo     https://github.com/venkatbilla2008/nlp-text-pipeline
    echo.
    echo [+] Next steps:
    echo     1. Go to https://share.streamlit.io
    echo     2. Find your app or click "New app"
    echo     3. It will auto-redeploy (1-2 minutes)
    echo     4. Test your app with translation!
    echo.
    echo [+] Your app URL will be:
    echo     https://venkatbilla2008-nlp-text-pipeline.streamlit.app
    echo.
    echo ==========================================
)

echo.
pause
