# PDFQuery

## Overview

**PDFQuery** is a FastAPI-based application that allows users to upload PDF documents, convert their content into vector embeddings using **ChromaDB**, and query the documents through a conversational chatbot powered by **Ollama LLM**. The chatbot uses **LangChain’s ConversationalRetrievalChain**, and if the answer cannot be found from the uploaded PDFs, it intelligently falls back to a **Wikipedia Agent** using LangChain tools to fetch answers from Wikipedia.

A user-friendly **Streamlit UI** is also included for easy interaction.

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

- Support for multiple file formats (e.g., DOCX, TXT).
- Enhanced file management dashboard.
- User authentication and session history.
- Add more knowledge sources (e.g.DOCX).
- Model selection dropdown in UI (e.g., choose between LLaMA, Mistral, etc.).

