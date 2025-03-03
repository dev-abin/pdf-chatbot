from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from typing import List, Tuple

app = FastAPI()

# Setup storage directories
STORAGE_DIR = r""
PDF_DIR = os.path.join(STORAGE_DIR, "pdfs")
VECTOR_DIR = os.path.join(STORAGE_DIR, "vectors")
PREF_MODEL = ""

# Create directories if they don't exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process the PDF document if not already processed."""
    try:
        print("Upload PDF endpoint called.")

        # Check if the uploaded file is a PDF
        if not file.filename.endswith('.pdf'):
            # Return error if file is not a PDF
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        content = await file.read() # Read the file content asynchronously
        file_hash = hashlib.sha256(content).hexdigest() # Generate a unique hash for the file
        pdf_path = os.path.join(PDF_DIR, f"{file_hash}.pdf") # Define the path to save the file
        
        # Check if the PDF has already been processed
        if os.path.exists(pdf_path):
            print({"message": f"{file.filename} has already been processed."})
            return {"message": f"{file.filename} has already been processed."} # Return a message without re-processing
        
        # Save the PDF file to the specified directory
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(content)
        
        # Initialize the vector store with embeddings
        vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=OllamaEmbeddings(model=PREF_MODEL))
        # Load the PDF content using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load() # Extract text from the PDF
        # Split the extracted text into smaller chunks for better embedding performance
        docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
        
        # Add the processed documents into the vector store
        vectorstore.add_documents(docs)
        print({"message": f"Successfully processed {file.filename}"})
        return {"message": f"Successfully processed {file.filename}"}
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Define a request model for chat input
class ChatRequest(BaseModel):
    question: str   # User's question
    chat_history: List[Tuple[str, str]] = []    # Chat history as a list of question-answer pairs


# Define a FastAPI endpoint for handling chat queries
@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    try:
        print("Chat endpoint called.")
        # Check if the vector store directory exists and is not empty
        if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
            raise HTTPException(status_code=404, detail="Vectorstore not found or empty. Please upload a PDF first.")
        
        # Load the vector store with the preprocessed document embeddings
        vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=OllamaEmbeddings(model=PREF_MODEL))

        # Create a conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=OllamaLLM(model=PREF_MODEL),    # Load the LLM model
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),     # Retrieve top 3 relevant documents
            return_source_documents=True    # Include source documents in the response
        )
        
        # Invoke the conversational retrieval chain with user input and chat history
        result = qa_chain.invoke({"question": chat_request.question, "chat_history": chat_request.chat_history})

        # Extract the generated answer from the response
        answer = result.get("answer", "No answer found.")

        # Extract source document metadata (page numbers) if available
        sources = [f"Page {doc.metadata.get('page', 'unknown')}" for doc in result.get("source_documents", [])]
        
        # Return the chatbot response along with the sources
        return {"answer": answer, "sources": sources}
    
    except Exception as e:
        print(f"Error in chat processing: {str(e)}")    # Log the error for debugging
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")    # Return a server error response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)
