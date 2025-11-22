@echo off


REM 1) Start Ollama server in a new command window
start cmd /k "ollama serve"

REM 2) Run Streamlit frontend (src/frontend/ui.py) with venv
start cmd /k "call ..\chatenv\Scripts\activate && python -m streamlit run apps\frontend\ui.py"

REM 3) Run FastAPI backend (backend.app:app) with venv
start cmd /k "call ..\chatenv\Scripts\activate && cd apps\backend\src && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

REM 4) Real-time view for backend logs
set LOGFILE=%~dp0apps\backend\logs\app.log
start powershell -NoExit -Command "Get-Content -Path '%LOGFILE%' -Tail 0 -Wait"