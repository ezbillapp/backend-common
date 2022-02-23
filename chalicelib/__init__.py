import logging
import os

_logger = logging.getLogger(__name__)

handler = logging.StreamHandler()
_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
_logger.addHandler(handler)
