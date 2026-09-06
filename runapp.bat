@echo off
setlocal

set ROOT=%~dp0

REM Ensure logs directory exists
set LOGDIR=%ROOT%apps\api\logs
if not exist "%LOGDIR%" (
    echo Creating logs directory: %LOGDIR%
    mkdir "%LOGDIR%"
)

REM 1) Start Ollama
start "" cmd /k "ollama serve"

REM 2) Frontend (Streamlit)
start "" cmd /k "cd /d %ROOT%apps\web && uv run streamlit run ui.py"

REM 3) Backend (FastAPI)
start "" cmd /k "cd /d %ROOT%apps\api && uv run uvicorn documind.main:app --host 0.0.0.0 --port 8000 --reload"

REM 4) Tail backend logs
set LOGFILE=%LOGDIR%\app.log
start "" powershell -NoExit -Command "Get-Content '%LOGFILE%' -Tail 0 -Wait"

endlocal
