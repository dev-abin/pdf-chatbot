@echo off
setlocal

set ROOT=%~dp0

REM Ensure logs directory exists
set LOGDIR=%ROOT%apps\backend\logs
if not exist "%LOGDIR%" (
    echo Creating logs directory: %LOGDIR%
    mkdir "%LOGDIR%"
)

REM 1) Start Ollama
start "" cmd /k "ollama serve"

REM 2) Frontend (Streamlit)
start "" cmd /k "cd /d %ROOT%apps\frontend && uv run streamlit run ui.py"

REM 3) Backend (FastAPI)
start "" cmd /k "cd /d %ROOT%apps\backend\src && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

REM 4) Tail backend logs
set LOGFILE=%LOGDIR%\app.log
start "" powershell -NoExit -Command "Get-Content '%LOGFILE%' -Tail 0 -Wait"

endlocal
