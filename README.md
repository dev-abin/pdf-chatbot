# PDFQuery

## Overview

**PDFQuery** is an AI-powered chatbot app that lets you upload PDF documents, turn their content into vector embeddings using ChromaDB, and ask questions in natural language through a conversational interface.

Built with FastAPI and Streamlit, it uses LangChain's ConversationalRetrievalChain powered by Ollama LLM to give smart, context-aware answers from your uploaded files. If the answer isn't found in your documents, it automatically falls back to a Wikipedia-powered agent to bring in relevant external knowledge.

---


## Features

- Upload and parse PDF documents.
- Store document embeddings in a **ChromaDB** vector store.
- Prevent reprocessing of already uploaded PDFs.
- Perform **contextual Q&A** via ConversationalRetrievalChain.
- **Fallback to Wikipedia Agent** if answer not found in documents.
- Display source page references for traceability.
- Interactive chatbot interface using **Streamlit**.

---

## Technologies Used

- **FastAPI** – Web framework for backend APIs.
- **ChromaDB** – Vector store for document embeddings.
- **Ollama LLM** – Language model to generate responses.
- **LangChain** – Framework for chaining LLMs, including tools and agents.
- **Wikipedia Tool** – LangChain tool for searching Wikipedia.
- **PyPDFLoader** – Extracts and preprocesses PDF content.
- **RecursiveCharacterTextSplitter** – Splits content for embedding.
- **Pydantic** – Data validation for API requests.
- **Streamlit** – UI for user interaction.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fastapi-pdf-chat.git
   cd fastapi-pdf-chat
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv chatenv
   source chatenv/bin/activate  # On Windows use `chatenv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Ensure that **ChromaDB** is set up properly and accessible.
4. Ensure ollama is running as a server:
   ```bash
   ollama serve
   ```
5. Start the FastAPI server:
   ```bash
   uvicorn chatapp:app --host 0.0.0.0 --port 8000 --reload
   ```
6. Run the Streamlit UI:
   ```bash
   streamlit run client_ui.py
   ```

## Run with Docker

You can easily run the entire application (FastAPI backend + Streamlit frontend) using Docker.

### 1. Build the Docker image

```bash
docker build -t pdfquery/chatbot:1.0 .
```

### 2. Run Ollama Server

Open a new terminal and run
```bash
ollama serve
```

### 3. Run the Docker container

```bash
docker run -e OLLAMA_API_URL=http://host.docker.internal:11434 -p 8000:8000 -p 8501:8501 pdfquery/chatbot:1.0
```

## API Endpoints
### Upload PDF
- **Endpoint:** `POST /upload-pdf/`
- **Description:** Uploads and processes a PDF file.
- **Request:**
  ```bash
  curl -X 'POST' \
    'http://localhost:8000/upload-pdf/' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@example.pdf'
  ```
- **Response:**
  ```json
  {"message": "Successfully processed example.pdf"}
  ```

### Chat with PDF Content
- **Endpoint:** `POST /chat/`
- **Description:** Ask a question based on the uploaded PDFs.
- **Request Body:**
  ```json
  {
    "question": "What is the main topic of the document?",
    "chat_history": []
  }
  ```
- **Response:**
  ```json
  {
    "answer": "The document discusses AI and machine learning trends.",
    "sources": ["Page 1", "Page 3"]
  }
  ```

---

## Wikipedia Agent Fallback

When no relevant answer is found in the PDF documents:

- The system uses **LangChain's Agent with Tools** to search Wikipedia.
- It follows a **step-by-step ReAct format** to ensure traceable reasoning:
  - **Thought → Action → Observation → Final Answer**

## Streamlit Interface
### Running the Chatbot UI
1. Start the FastAPI server as mentioned earlier.
2. Run the Streamlit app using:
   ```bash
   streamlit run client_ui.py
   ```
3. The web interface will allow you to:
   - Upload a PDF document.
   - Ask questions related to the uploaded document.
   - View responses along with source page references.

## Error Handling
- If a non-PDF file is uploaded, the API returns:
  ```json
  {"detail": "File must be a PDF"}
  ```
- If no PDFs are processed before querying:
  ```json
  {"detail": "Vectorstore not found or empty. Please upload a PDF first."}
  ```
- Internal errors return:
  ```json
  {"detail": "Internal server error: <error_message>"}
  ```
---

## Future Enhancements

- Support for multiple file formats (e.g., DOCX, TXT, CSV).
- Enhanced file management dashboard.
- User authentication and session history.
- Model selection dropdown in UI (e.g., choose between LLaMA, Mistral, etc.).

