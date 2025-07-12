
REM Start Ollama server in a new command window
start cmd /k ollama serve

REM Run client_ui.py in a new command window (with venv activated)
start cmd /k "call ..\chatenv\Scripts\activate && python -m streamlit run app\client_ui.py"

REM Run chatapp.py in another new command window (with venv activated)
start cmd /k "call ..\chatenv\Scripts\activate && python -m uvicorn app.chatapp:app --host 0.0.0.0 --port 8000 --reload"

REM Real-time view for app log
start powershell -NoExit -Command "Get-Content -Path 'app\logs\app.log' -Tail 0 -Wait"

