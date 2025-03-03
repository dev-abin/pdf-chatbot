
REM Run client_ui in a new command window
start cmd /k "python -m streamlit run client_ui.py"

REM Run chatapp.py in another new command window
start cmd /k "python -m uvicorn chatapp:app --host 0.0.0.0 --port 8000 --reload"


