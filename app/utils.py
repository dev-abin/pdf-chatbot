
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import OllamaLLM
import json
import uuid
from datetime import datetime, timezone
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain.schema import Document
from app.logger import rag_logger, logger
from app.config import OLLAMA_API_URL,  PREF_MODEL, PREF_EMBEDDING_MODEL


def preprocess_file_content(raw_docs):
    """
    This function performs several text cleaning operations on file content:
    Extracts text from all pages of the file
    Normalizes text by fixing hyphenation, standardizing Unicode, and removing special characters
    Standardizes whitespace and paragraph formatting
    
    Returns:
        List of document chunks with cleaned text content
    """

    import re
    import unicodedata
    import regex  # For Unicode property support beyond what re offers

    # Create list to store cleaned documents
    cleaned_docs = []

    logger.info("preprocessing file content")

    try:
        for doc in raw_docs:
            # Extract text content from the document
            text = doc.page_content
            
            # Fix hyphenated words split across lines (e.g., "ex-\nample" becomes "example")
            # Handles variations in whitespace around hyphens
            text = re.sub(r'-\s*\n\s*', '', text)
            
            # Normalize Unicode for consistent character representation
            text = unicodedata.normalize('NFC', text)
            
            # Remove all characters that aren't letters, numbers, punctuation or whitespace
            # \p{L}=letters, \p{N}=numbers, \p{P}=punctuation, \s=whitespace
            text = regex.sub(r'[^\p{L}\p{N}\p{P}\s]', '', text)
            
            # Standardize paragraph breaks: replace 3+ newlines with exactly 2
            # Preserves intentional paragraph separation while removing excessive breaks
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Replace multiple spaces/tabs with a single space
            # Normalizes spacing within paragraphs
            text = re.sub(r'[ \t]+', ' ', text)
            
            # Remove leading and trailing whitespace from the entire document
            text = text.strip()
            
            # Update the document with cleaned text
            doc.page_content = text
            
            # Add to cleaned documents list
            cleaned_docs.append(doc)
    except Exception:
        # Catch any errors, log them, and return an empty list
        logger.exception("preprocess_file_content failed")
        return []

    return cleaned_docs


def extract_pdf_content_ocr(pdf_path, tesseract_config='--psm 6', zoom=3):
    """
    Extract text content from a PDF file using OCR only.
    
    Args:
        pdf_path (str): Path to the PDF file
        tesseract_config (str): Tesseract configuration string (default: '--psm 6')
        zoom (int): Zoom factor for image resolution (default: 3, higher = better quality)
    
    Returns:
        list: List of Document objects compatible with LangChain (same format as PyPDFLoader)
    """
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        documents = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Convert page to image with high resolution for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            ocr_text = pytesseract.image_to_string(img, config=tesseract_config)
            
            # Create Document object with metadata (same format as PyPDFLoader)
            document = Document(
                page_content=ocr_text,
                metadata={
                    "source": pdf_path,
                    "page": page_num
                }
            )
            documents.append(document)
        
        doc.close()
        return documents

    except Exception:
        # Catch any errors, log them, and return an empty list
        logger.exception("extract_pdf_content_ocr failed")
        return []


# === Wikipedia Fallback Agent ===
def initiate_wikipedia_agent():
    """Initialize a Wikipedia agent for answering general knowledge questions.

    This function sets up a LangChain agent that uses the Wikipedia API to retrieve information.
    The agent is configured with a custom prompt format to ensure structured responses, including
    the user's question, step-by-step reasoning, tool actions, observations, and a final answer.
    The agent uses the Ollama LLM as its language model and is set up for zero-shot reasoning.

    Returns:
        AgentExecutor: A configured LangChain agent ready to process queries using Wikipedia.
    """
    # Initialize the Wikipedia tool using the WikipediaAPIWrapper
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # Define a list of tools (only Wikipedia in this case) for the agent to use
    tools = [
        Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Search Wikipedia for general knowledge."
        )
    ]

    # custom prompt kwargs to enforce a structured response format
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
    "suffix": """Begin!\n\n
    Question: {input}\n
    {agent_scratchpad}
    Thought:"""
    }

    # Initialize and return the LangChain agent with the specified tools, LLM, and prompt
    return initialize_agent(
        tools=tools,
        llm=OllamaLLM(model=PREF_MODEL, base_url=OLLAMA_API_URL),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs=agent_kwargs,
        verbose=True,   # Enable detailed logging for debugging
        handle_parsing_errors=True  # Handle parsing errors gracefully
    )


def log_interaction(query, contexts, llm_response, ground_truth=None):
    """Log user interaction details for a Retrieval-Augmented Generation (RAG) query.

    This function creates a structured log entry for a user query, including the query text,
    retrieved contexts, LLM response, and optional ground truth. The log entry is assigned a
    unique ID and timestamp, and is written to the logger in JSON format.

    Args:
        query (str): The user's query or question.
        contexts (list): A list of retrieved context strings used to generate the response.
        llm_response (str): The response generated by the language model.
        ground_truth (str, optional): The expected or correct response, if available. Defaults to None.

    """
    # Creates a dictionary to store interaction details
    interaction = {
        "interaction_id": str(uuid.uuid4()),                 # Generate a unique ID for the interaction
        "timestamp": datetime.now(timezone.utc).isoformat(), # Record the current UTC timestamp
        "query": query,                                      # Store the user's query
        "contexts": contexts,                                # Store the retrieved contexts
        "llm_response": llm_response,                        # Store the LLM's response
        "ground_truth": ground_truth,                        # Store the ground truth, if provided
    }
    # Log the interaction as a JSON string using the configured logger
    rag_logger.info(json.dumps(interaction))
    return True


# Load logged data
def load_logged_interactions(filepath):
    """Load logged interaction records from a JSON log file.

    This function reads a log file containing JSON-formatted interaction records generated 
    by the Retrieval-Augmented Generation (RAG) system. Each line in the file is
    expected to be a JSON object, nested  with a 'message' field containing the
    interaction data. The function extracts relevant fields (query, response, contexts, and
    ground truth) and returns them as a list of structured dictionaries.

    Args:
        filepath (str): The path to the log file containing JSON-formatted interaction records.

    Returns:
        list: A list of dictionaries, each containing:
            - user_input (str): The user's query or question.
            - response (str): The response generated by the language model.
            - retrieved_contexts (list): A list of retrieved context strings used to generate the response.
            - reference (str or None): The ground truth or expected response, if available.
    """
    # Initialize an empty list to store the parsed records
    records = []

    try:
        # Open the log file in read mode
        with open(filepath, "r") as f:
            for line in f:
                # Parse the line as a JSON object
                raw = json.loads(line)

                # Check if the JSON object has a nested 'message' field
                if "message" in raw:
                    # Parse the nested 'message' field as another JSON object
                    data = json.loads(raw["message"])

                    # Create a structured record with relevant fields
                    record = {
                        "user_input": data["query"],
                        "response": data["llm_response"],
                        "retrieved_contexts": data["contexts"],
                        "reference": data["ground_truth"]
                    }

                    # Append the record to the list
                    records.append(record)

    except Exception:
        # Catch any errors, log them, and return an empty list
        logger.exception("load_logged_interactions failed")
        return []

    # Return the list of parsed records
    return records


def evaluate_rag_response():
    """Evaluate the performance of a RAG system using logged interaction data.

    This function loads the most recent interaction record from a JSONL log file and evaluates
    the Retrieval-Augmented Generation (RAG) system's performance using the `ragas` library.
    It assesses the response using metrics such as Faithfulness, Context Precision, and Response
    Relevancy, leveraging an Ollama LLM and HuggingFace embeddings. The evaluation results are
    logged, and the function returns True to indicate successful execution.

    Args:
        None: This function does not take any parameters

    Returns:
        bool: Returns True if the evaluation was successful, False if an error occurs
    """
    try:
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
        from ragas import EvaluationDataset
        from ragas.llms import LangchainLLMWrapper
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas import evaluate
        from ragas.run_config import RunConfig
        import os

        # Get absolute path to the directory of the current script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Construct full path to logs directory
        log_file_path = os.path.join(BASE_DIR, "logs", "rag_interactions.jsonl")
        
        # get the logged records
        records = load_logged_interactions(log_file_path)

        # Select only the most recent record for evaluation
        records = [records[-1]]

        # Create an EvaluationDataset from the selected record
        dataset = EvaluationDataset.from_list(records)

        # Initialize the evaluator LLM using Ollama with predefined model and API URL
        evaluator_llm = LangchainLLMWrapper(OllamaLLM(model=PREF_MODEL, base_url= OLLAMA_API_URL))

        # spcify the embedding fn
        embedding_fn = HuggingFaceEmbeddings(model_name=PREF_EMBEDDING_MODEL)

        # Wrap the embeddings for compatibility with ragas
        embedding   = LangchainEmbeddingsWrapper(embedding_fn)

        # Define the metrics for evaluation
        metrics = [
                LLMContextPrecisionWithoutReference(), # Measures precision of retrieved contexts
                Faithfulness(),                        # Assesses factual accuracy of the response
                ResponseRelevancy()                    # Evaluates relevance of the response to the query
            ]

        # Configure evaluation to run with 1 concurrent metric call and a 300-second timeout
        run_config = RunConfig(timeout=300, max_workers=1)

        # Perform the evaluation using the dataset, metrics, LLM, and embeddings
        results = evaluate(dataset=dataset,metrics= metrics, llm=evaluator_llm, embeddings=embedding, run_config=run_config)
        
        logger.info(results)

        return True
    
    except Exception:
            # Log any unexpected errors during evaluation and return False
            logger.exception("evaluate_rag_response failed")
            return False


if __name__=="__main__":
    evaluate_rag_response()
