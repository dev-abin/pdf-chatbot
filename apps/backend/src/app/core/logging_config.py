import json
import logging
import logging.config
from datetime import UTC, datetime

import yaml

from ..core.settings import LOG_DIR, LOGGING_CONFIG_PATH


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)

    yaml_path = LOGGING_CONFIG_PATH
    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["formatters"]["json_fmt"]["()"] = JsonFormatter

    config["handlers"]["app_rotating_file"]["filename"] = str(LOG_DIR / "app.log")
    config["handlers"]["rag_json_handler"]["filename"] = str(
        LOG_DIR / "rag_interactions.jsonl"
    )

    logging.config.dictConfig(config)


logger = logging.getLogger("pdf_query_bot")
rag_logger = logging.getLogger("rag_logger")
