import os
from pathlib import Path

# Get project folder dir
current_file = Path(__file__).resolve()
project_dir = current_file.parent.parent

ENV_PATH = project_dir / ".env"
LOG_PATH = project_dir / "logs"

INPUT_DATA_PATH = project_dir /"data/input_data2.json"
OUTPUT_DATA_PATH = project_dir /"data/output_data.jsonl"

HOST = os.getenv("HOST_DOCKER_INTERNAL", "127.0.0.1") #HOST_DOCKER_INTERNAL exists in Docker deployment

MILVUS_URL = f"http://{HOST}:19530"
EMBED_SERVICE_URL = f"http://{HOST}:8083"
MULTIMODEL_SERVICE_URL = f"http://{HOST}:8010"
BGM_SERVICE_URL = f"http://{HOST}:8011"
OUTPUT_REQUEST_URL = None

if __name__=="__main__":
    print(MILVUS_URL, EMBED_SERVICE_URL, MULTIMODEL_SERVICE_URL, BGM_SERVICE_URL)