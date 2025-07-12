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
from app.utils import ChatRequest, prompt_template, OLLAMA_API_URL,  PREF_MODEL, PREF_EMBEDDING_MODEL,\
    initiate_wikipedia_agent, log_interaction, preprocess_pdf_content, extract_pdf_content_ocr
from app.logger import logger

app = FastAPI()

# Setup storage directories
STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(STORAGE_DIR, "pdfs")
VECTOR_DIR = os.path.join(STORAGE_DIR, "vectors")

# a string literal for answer not found message
NO_ANSWER_FOUND = "No answer found"

# Create directories if they don't exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


# === Upload Endpoint ===
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF document if not already processed.
    
    Args:
        file (UploadFile): A required PDF file to be uploaded.

    Returns:
        dict: A dictionary containing the filename and a success message.

    Raises:
        HTTPException: If the uploaded file is not a PDF.
    """
    try:

        logger.info(f"Upload PDF endpoint called | endpoint=/upload-pdf/ | uploaded_pdf='{file.filename}'")

        # Check if the uploaded file is a PDF
        if not file.filename.endswith('.pdf'):
            # Return error if file is not a PDF
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        content = await file.read()                          # Read the file content asynchronously
        file_hash = hashlib.sha256(content).hexdigest()      # Generate a unique hash for the file
        pdf_path = os.path.join(PDF_DIR, f"{file_hash}.pdf") # Define the path to save the file
        
        # Check if the PDF has already been processed
        if os.path.exists(pdf_path):
            logger.info(f"{file.filename} has already been processed.")
            return {"message": f"{file.filename} has already been processed."} # Return a message without re-processing
        
        logger.info("storing the pdf contents to vector store")

        # Save the PDF file to the specified directory
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(content)

        # define the embedding_fn
        embedding_fn = HuggingFaceEmbeddings(
                model_name=PREF_EMBEDDING_MODEL
            )
                    
        # Initialize the vector store with embeddings
        vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_fn)

        # Load the PDF content using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load() # Extract text from the PDF
        
        if not raw_docs or all(doc.page_content.strip() == "" for doc in raw_docs):
            logger.info("No extractable text found. The PDF may be scanned or image-based. Proceeding to do OCR")
            raw_docs = extract_pdf_content_ocr(pdf_path)
        
        # preprocess the pdf content
        cleaned_docs = preprocess_pdf_content(raw_docs)
        
        # Split the extracted text into smaller chunks for better embedding performance
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(cleaned_docs)
        
        # Add the processed documents into the vector store
        vectorstore.add_documents(docs)

        logger.info(f"Successfully processed {file.filename}")
        return {"message": f"Successfully processed {file.filename}"}
    
    except Exception as e:
        logger.exception(f"{file.filename} processing failed")

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# === Chat Endpoint ===
@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    """Handle user chat queries by retrieving relevant information from a vector store or falling back to a Wikipedia agent.
    
    This endpoint processes a user's question by querying a preloaded vector store containing document embeddings.
    If the vector store is empty or unavailable, it raises an error. If the query cannot be answered using the vector store,
    it falls back to a Wikipedia agent. The response includes the generated answer and sources (either document pages or Wikipedia).

    Args:
        chat_request (ChatRequest): A request object containing the user's question and chat history.

    Returns:
        dict: A dictionary containing:
            - answer (str): The generated answer to the user's question.
            - sources (list): A list of sources used to generate the answer (e.g., document page numbers or "Wikipedia").

    Raises:
        HTTPException:
            - 404: If the vector store directory is not found or empty (indicating no PDF has been uploaded).
    """
    try:

        logger.info(f"Chat endpoint called. | endpoint=/chat/ | query='{chat_request.question}' | chat_history= '{chat_request.chat_history}'")

        # Check if the vector store directory exists and is not empty
        if not os.path.exists(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
            raise HTTPException(status_code=404, detail="Vectorstore not found or empty. Please upload a PDF first.")
        
        # define the embedding_fn
        embedding_fn = HuggingFaceEmbeddings(
                model_name=PREF_EMBEDDING_MODEL
            )
        
        # Load the vector store with the preprocessed document embeddings
        vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_fn)

        # Create PromptTemplate
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )
    
        # Create a conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=OllamaLLM(model=PREF_MODEL, base_url= OLLAMA_API_URL),    # Load the LLM model
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),   # Retrieve top 3 relevant documents
            return_source_documents=True,                                 # Include source documents in the response
            combine_docs_chain_kwargs={"prompt": prompt},                 # apply custom prompt here
            verbose = True
        )
        
        # Invoke the conversational retrieval chain with user input and chat history
        result = qa_chain.invoke({"question": chat_request.question, "chat_history": chat_request.chat_history})

        # Extract the generated answer from the response
        answer = result.get("answer", NO_ANSWER_FOUND).strip()
        logger.info(answer)

        if NO_ANSWER_FOUND in answer:
            logger.info(f"Falling back to Wikipedia agent | query='{chat_request.question}' | chat_history= '{chat_request.chat_history}'")

            agent = initiate_wikipedia_agent()
            # invoke the agent with the user query
            agent_result = agent.invoke(chat_request.question)
            # get the answer from wikipedia agent
            agent_answer = agent_result.get("output", NO_ANSWER_FOUND)
            
            return {"answer": agent_answer, "sources": ["Wikipedia"]}
        
        retrieved_contexts = [ doc.page_content for doc in result["source_documents"]]

        # log user interaction with llm
        log_interaction(chat_request.question, retrieved_contexts, answer)

        sources = [f"Page {doc.metadata.get('page', 'unknown')}" for doc in result.get("source_documents", [])]
        # Return the chatbot response along with the sources
        return {"answer": answer, "sources": sources}

    except Exception as e:
        logger.exception("Processing chat query failed")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")    # Return a server error response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)
