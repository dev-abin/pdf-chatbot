import logging
import os
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime, timezone

# Get absolute path to the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# path to log folder
log_folder = os.path.join(BASE_DIR, "logs")

# Create logs folder if it doesn't exist
os.makedirs(log_folder, exist_ok=True)

# Construct full path to logs directory
app_log_file_path = os.path.join(log_folder, "app.log")
rag_log_file_path = os.path.join(log_folder, "rag_interactions.jsonl")

# Create logger instance named "pdf_query_bot"
logger = logging.getLogger("pdf_query_bot")
# Set the minimum logging level for this logger (DEBUG captures everything)
logger.setLevel(logging.DEBUG)

# Formatter with timestamp, level, filename, line number, and message
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
)

# Create a rotating file handler:
# - Logs will go to 'logs/app.log'
# - Each log file is limited to 5 MB
# - Keeps up to 3 backup files (app.log.1, app.log.2, etc.)
file_handler = RotatingFileHandler(app_log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3)
# Apply the formatter to the file handler
file_handler.setFormatter(formatter)
# Set the log level for the file handler to DEBUG (logs everything)
file_handler.setLevel(logging.DEBUG)


# Add handlers
logger.addHandler(file_handler)


# Custom JSON formatter
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)

# Setup the logger
rag_logger = logging.getLogger("rag_logger")
rag_logger.setLevel(logging.INFO)

rag_file_handler = logging.FileHandler(rag_log_file_path)  # JSONL format
rag_file_handler.setFormatter(JsonFormatter())

rag_logger.addHandler(rag_file_handler)

