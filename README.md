# 🎬 AI Multimedia Material Processing Service

> Automatically parses video, image, and audio using multimodal LLMs to generate structured summaries, embed them as vectors, and persist them in Milvus. Supports real-time SSE streaming for seamless progress and result synchronization with business systems.

[📖 Full API Documentation](./API_doc.md)

---

## ✨ Key Features

-  **Multimodal Parsing**: Automatically extracts semantic content from videos, images, and audio to generate structured descriptions.
- 🔢 **Vector Retrieval**: Embeds summaries into 1024-dimensional vectors for efficient semantic search and matching.
- 🗄️ **Milvus Integration**: Automatically manages collections, HNSW indexes, and partition storage by material type.
- 📡 **Real-time SSE Streaming**: Pushes progress updates and individual results during processing, enabling live frontend updates.
- 🔒 **Single-Instance Serialization**: Global async lock ensures only one task processes at a time, keeping resource usage predictable.
- 🔄 **Dual Output Channels**: Results are simultaneously written to local JSONL files and optionally pushed via HTTP callbacks.

---

## 🏗️ Data Processing Flow

1. **Request Handling** → `POST /stream_process`
2. **FastAPI Server** → Async locking | SSE stream management
3. **Multimodal Pipeline** → Video/Image/Audio → LLM Summarization → Vector Embedding
4. **Milvus Storage** → Collection management | HNSW indexing | Partitioned storage
5. **Result Distribution** → Local JSONL archiving | External HTTP callbacks | Real-time SSE push

---

## 🚀 Quick Start

### 📦 Prerequisites
- Python 3.11+
- Docker & Docker Compose (recommended for production)
- Milvus 2.6.6+
- Dependent Services:
  - `multimodal_service` (Qwen3.5-9B): Video & image summarization
  - `bgm_service` (Qwen-Audio-Chat): Audio summarization
  - `embed_service` (BGE-Large): Text vectorization

###  Local Development
```bash
# 1. Clone and navigate to the project
git clone http://gogs.km360.cn/lituo/process-media.git
cd process-media

# 2. Install dependencies
uv sync

# 3. Configure settings
# Edit config/path_config.py and config/constant_config.py to set service URLs and paths

# 4. Start the service
uvicorn video_director:app --host 0.0.0.0 --port 8013 --reload
```