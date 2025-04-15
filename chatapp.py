from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
import utils
from logger import logger

app = FastAPI()

# Setup storage directories
STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(STORAGE_DIR, "pdfs")
VECTOR_DIR = os.path.join(STORAGE_DIR, "vectors")

# Create directories if they don't exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process the PDF document if not already processed."""
    try:

        logger.info(f"Upload PDF endpoint called | endpoint=/upload-pdf/ | uploaded_pdf='{file.filename}'")

        # Check if the uploaded file is a PDF
        if not file.filename.endswith('.pdf'):
            # Return error if file is not a PDF
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        content = await file.read() # Read the file content asynchronously
        file_hash = hashlib.sha256(content).hexdigest() # Generate a unique hash for the file
        pdf_path = os.path.join(PDF_DIR, f"{file_hash}.pdf") # Define the path to save the file
        
        # Check if the PDF has already been processed
        if os.path.exists(pdf_path):
            logger.info(f"{file.filename} has already been processed.")
            return {"message": f"{file.filename} has already been processed."} # Return a message without re-processing
        
        # Save the PDF file to the specified directory
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(content)

        embedding_fn = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
                    
        # Initialize the vector store with embeddings
        vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_fn)

        # Load the PDF content using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load() # Extract text from the PDF

        # Split the extracted text into smaller chunks for better embedding performance
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        
        # Add the processed documents into the vector store
        vectorstore.add_documents(docs)

        logger.info(f"Successfully processed {file.filename}")
        return {"message": f"Successfully processed {file.filename}"}
    
    except Exception as e:
        logger.exception(f"{file.filename} processing failed")

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# === Main Chat Endpoint ===
@app.post("/chat/")
async def chat(chat_request: utils.ChatRequest):
    try:

        logger.info(f"Chat endpoint called. | endpoint=/chat/ | query='{chat_request.question}' | chat_history= '{chat_request.chat_history}'")

        # Check if the vector store directory exists and is not empty
        if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
            raise HTTPException(status_code=404, detail="Vectorstore not found or empty. Please upload a PDF first.")
        
        embedding_fn = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Load the vector store with the preprocessed document embeddings
        vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_fn)

        # Create PromptTemplate
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=utils.prompt_template

        )
    
        # Create a conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=OllamaLLM(model=utils.PREF_MODEL, base_url= utils.OLLAMA_API_URL),    # Load the LLM model
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),     # Retrieve top 3 relevant documents
            return_source_documents=True,    # Include source documents in the response
            combine_docs_chain_kwargs={"prompt": prompt},  # apply custom prompt here
            verbose = True
        )
        
        # Invoke the conversational retrieval chain with user input and chat history
        result = qa_chain.invoke({"question": chat_request.question, "chat_history": chat_request.chat_history})

        # Extract the generated answer from the response
        answer = result.get("answer", "No answer found").strip()

        # logger.info(answer)

        if "No answer found" in answer:

            logger.info(f"Falling back to Wikipedia agent | query='{chat_request.question}' | chat_history= '{chat_request.chat_history}'")

            agent = utils.get_wikipedia_agent()
            agent_result = agent.invoke(chat_request.question)
            agent_answer = agent_result.get("output", "No answer found")
            
            return {"answer": agent_answer, "sources": ["Wikipedia"]}

        sources = [f"Page {doc.metadata.get('page', 'unknown')}" for doc in result.get("source_documents", [])]
        # Return the chatbot response along with the sources
        return {"answer": answer, "sources": sources}

    except Exception as e:
        logger.exception(f"Processing chat query failed")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")    # Return a server error response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)
