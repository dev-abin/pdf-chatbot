import os

# Setup storage directories
STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(STORAGE_DIR, "files")
VECTOR_DIR = os.path.join(STORAGE_DIR, "vectors")

# Create directories if they don't exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(FILE_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# This checks for the environment variable `OLLAMA_API_URL`.
# If it's not set, it defaults to localhost (for local dev).
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

PREF_MODEL = "llama3.1:8b"
PREF_EMBEDDING_MODEL = "sentence-transformers/msmarco-distilbert-base-v3"
# PREF_EMBEDDING_MODEL = "BAAI/bge-en-icl"
FILE_EXTENSIONS = (".pdf", ".docx", ".txt")

# a string literal for answer not found message
NO_ANSWER_FOUND = "No answer found"

# Define custom prompt
GENERATION_PROMPT = """
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