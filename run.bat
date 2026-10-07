@echo off
setlocal
cd /d "%~dp0"

if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
) else (
    where uv >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv was not found. Install uv, then run this file again.
        pause
        exit /b 1
    )
    set "UV_EXE=uv"
)

if not exist ".env" (
    echo ERROR: .env was not found. Add OPENAI_API_KEY, then run this file again.
    pause
    exit /b 1
)

echo Starting Streamlit chatbot. Press Ctrl+C in this window to stop it.
"%UV_EXE%" run streamlit run app.py

if errorlevel 1 pause
