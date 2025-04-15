import logging
import os
from logging.handlers import RotatingFileHandler

# Create logs folder if it doesn't exist
os.makedirs("logs", exist_ok=True)

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
file_handler = RotatingFileHandler(
    "logs/app.log", maxBytes=5 * 1024 * 1024, backupCount=3
)
# Apply the formatter to the file handler
file_handler.setFormatter(formatter)
# Set the log level for the file handler to DEBUG (logs everything)
file_handler.setLevel(logging.DEBUG)


# Add handlers
logger.addHandler(file_handler)
