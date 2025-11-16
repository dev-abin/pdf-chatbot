@echo off

REM ------------------------------------------------------------
REM Ensure we run from the project root (folder of this .bat)
REM ------------------------------------------------------------
REM cd /d "%~dp0"

REM ============================================================
REM 1) Start Ollama server in a new command window
REM ============================================================
start cmd /k "ollama serve"

REM ============================================================
REM 2) Run Streamlit frontend (src/frontend/ui.py) with venv
REM    - venv is at: C:\PDFQuery\chatenv
REM ============================================================
start cmd /k "call ..\chatenv\Scripts\activate && python -m streamlit run src\frontend\ui.py"

REM ============================================================
REM 3) Run FastAPI backend (backend.app:app) with venv
REM    - cd into src so 'backend.app' is importable
REM ============================================================
start cmd /k "call ..\chatenv\Scripts\activate && cd src && python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"

REM ============================================================
REM 4) Real-time view for app log (logs\app.log)
REM ============================================================
start powershell -NoExit -Command "Set-Location '%cd%'; Get-Content -Path 'logs\app.log' -Tail 0 -Wait"
