import logging
import datetime
import os
from config.path_config import LOG_PATH

# Create log folder if it doesn't exist
os.makedirs(LOG_PATH, exist_ok=True)

# Generate a timestamp for the log filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

process_media_log_filename = os.path.join(LOG_PATH, f"process_media_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(process_media_log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
process_media_logger = logging.getLogger(__name__)