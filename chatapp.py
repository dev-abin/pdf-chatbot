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
from pydantic import BaseModel
from typing import List, Tuple
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

app = FastAPI()

# Setup storage directories
STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(STORAGE_DIR, "pdfs")
VECTOR_DIR = os.path.join(STORAGE_DIR, "vectors")
PREF_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434/" 

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

        embedding_fn = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
                    
        # Initialize the vector store with embeddings
        vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_fn)
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


# Define custom prompt
custom_prompt_template = """
You are an AI assistant. Use only the information provided in the context to answer the user's question.

Context:
{context}

Question: {question}

Rules:
- If the context does not contain information relevant to the question, respond strictly with: "No answer found"
- Do not make assumptions.
- Do not try to be helpful beyond the context.
- Do not guess or provide partial answers.

Answer:
"""


# === Wikipedia Fallback Agent ===
def get_wikipedia_agent():
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [
        Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Search Wikipedia for general knowledge."
        )
    ]

    # Custom system message to enforce format
    agent_kwargs = {
        "prefix": """You are a helpful AI assistant.
You can use tools to look up relevant information.
Always return answers in this format:

Question: <user question>
Thought: <step-by-step reasoning>
Action: <tool name>(<input>)
Observation: <tool result>
... (repeat Thought/Action/Observation as needed)
Final Answer: <your final answer to the user>""",
        "suffix": """Begin!\n\nQuestion: {input}\nThought:"""
    }

    return initialize_agent(
        tools=tools,
        llm=OllamaLLM(model=PREF_MODEL, base_url=OLLAMA_BASE_URL),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs=agent_kwargs,
        verbose=True,
        handle_parsing_errors=True
    )


# === Main Chat Endpoint ===
@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    try:
        print("Chat endpoint called.")
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
            template=custom_prompt_template

        )
    
        # Create a conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=OllamaLLM(model=PREF_MODEL, base_url= OLLAMA_BASE_URL),    # Load the LLM model
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),     # Retrieve top 3 relevant documents
            return_source_documents=True,    # Include source documents in the response
            combine_docs_chain_kwargs={"prompt": prompt},  # apply custom prompt here
            verbose = True
        )
        
        # Invoke the conversational retrieval chain with user input and chat history
        result = qa_chain.invoke({"question": chat_request.question, "chat_history": chat_request.chat_history})

        # Extract the generated answer from the response
        answer = result.get("answer", "No answer found").strip()

        if "No answer found" in answer:
            print("Falling back to Wikipedia agent...")

            agent = get_wikipedia_agent()
            agent_result = agent.invoke(chat_request.question)
            agent_answer = agent_result.get("output", "No answer found")
            
            return {"answer": agent_answer, "sources": ["Wikipedia"]}

        sources = [f"Page {doc.metadata.get('page', 'unknown')}" for doc in result.get("source_documents", [])]
        # Return the chatbot response along with the sources
        return {"answer": answer, "sources": sources}

    except Exception as e:
        print(f"Error in chat processing: {str(e)}")    # Log the error for debugging
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")    # Return a server error response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True)
