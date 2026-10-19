import json
import logging
import sys
from datetime import datetime, timezone

class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }

        fields = (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "error_type",
        )

        for field in fields:
            value = getattr(record,field, None)
            if value is not None:
                data[field] = value

        return json.dumps(data)


def configure_logging():
    logger = logging.getLogger("workflow_ai")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    logger.propagate = False

    return logger