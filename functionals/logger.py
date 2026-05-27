import logging
import datetime
import os
import re
import threading
import time
from config.constant_config import LOG_KEEPING_HOURS, LOG_PREFIX
from config.path_config import LOG_PATH

# Create log folder if it doesn't exist
os.makedirs(LOG_PATH, exist_ok=True)

# Generate a timestamp for the log filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
process_media_log_filename = os.path.join(LOG_PATH, f"{LOG_PREFIX}_{timestamp}.log")

# Safeguard: Store the active filename to prevent deleting the file currently being written to
active_log_basename = os.path.basename(process_media_log_filename)

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

def cleanup_old_logs():
    """Runs in a background daemon thread. Deletes logs >24h by parsing filename timestamp."""
    while True:
        try:
            now = datetime.datetime.now()
            for filename in os.listdir(LOG_PATH):
                # SAFEGUARD: NEVER delete the currently active log file
                if filename == active_log_basename:
                    continue

                # Strict regex matching your exact naming convention
                pattern = rf'^{re.escape(LOG_PREFIX)}_(\d{{8}}_\d{{6}})\.log$'
                match = re.search(pattern, filename)
                if match:
                    file_time_str = match.group(1)
                    try:
                        file_time = datetime.datetime.strptime(file_time_str, "%Y%m%d_%H%M%S")
                        if (now - file_time) > datetime.timedelta(hours=LOG_KEEPING_HOURS):
                            filepath = os.path.join(LOG_PATH, filename)
                            os.remove(filepath)
                            process_media_logger.info(f"[日志清除] 已删除旧日志: {filename}")
                    except ValueError:
                        continue  # Skip malformed filenames silently
        except FileNotFoundError:
            # Log dir might be temporarily unavailable during startup/remounts
            time.sleep(60)
            continue
        except Exception as e:
            # exc_info=True provides the full stack trace in the logs
            process_media_logger.error(f"[日志清除] 报错: {e}", exc_info=True)

        time.sleep(3600)  # Check hourly

# Start cleanup thread AFTER logger is ready
# daemon=True ensures this thread automatically dies when the main app/container stops
cleanup_thread = threading.Thread(target=cleanup_old_logs, daemon=True)
cleanup_thread.start()