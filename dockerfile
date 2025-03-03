# Use an official Python image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose necessary ports
EXPOSE 8000 8501

# Run both FastAPI and Streamlit
CMD uvicorn chatapp:app --host 0.0.0.0 --port 8000 & streamlit run client_ui.py --server.port 8501 --server.address 0.0.0.0
