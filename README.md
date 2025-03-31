# PDFQuery

## Overview
This project is a FastAPI-based application that allows users to upload PDF documents, process their content into a vector database, and interact with the stored knowledge using a conversational retrieval-based approach. It leverages **ChromaDB** for vector storage and **Ollama LLM** for answering user queries. A **Streamlit UI** is provided for an interactive chatbot experience.

## Features
- Upload PDF files and store their content for retrieval.
- Prevent duplicate processing of already uploaded PDFs.
- Store processed text embeddings in a vector database using **ChromaDB**.
- Perform conversational queries with **ConversationalRetrievalChain**.
- Generate responses with **Ollama LLM** based on the stored knowledge.
- **Streamlit UI** for an interactive chatbot experience.

## Technologies Used
- **FastAPI** - Web framework for building the API.
- **ChromaDB** - Vector database for storing document embeddings.
- **Ollama LLM** - Language model for answering queries.
- **PyPDFLoader** - Extracts text from PDF documents.
- **RecursiveCharacterTextSplitter** - Splits documents into manageable chunks for vector storage.
- **Pydantic** - Data validation and serialization.
- **Streamlit** - Web-based UI for interacting with the chatbot.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fastapi-pdf-chat.git
   cd fastapi-pdf-chat
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Ensure that **ChromaDB** is set up properly and accessible.
4. Start the FastAPI server:
   ```bash
   uvicorn chatapp:app --host 0.0.0.0 --port 8000 --reload
   ```
5. Run the Streamlit UI:
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

## Future Enhancements
- Support for multiple document formats.
- Integration with a frontend for better usability.
- Improved model selection for better response accuracy.

