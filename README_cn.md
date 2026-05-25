# 🎬 AI 多媒体材料处理服务

> 基于多模态大模型自动解析视频、图片与音频，生成结构化摘要并嵌入向量，持久化存储至 Milvus。支持 SSE 实时流式输出，便于业务系统同步进度与结果。

[📖 完整 API 文档](./API_doc.md)

---

## ✨ 核心功能

- 🔍 **多模态解析**：自动提取视频/图片/音频语义，生成结构化描述
- 🔢 **向量检索**：摘要文本嵌入为 1024 维向量，支持高效语义匹配
- 🗄️ **Milvus 集成**：自动管理 Collection、HNSW 索引与按类型分区
- 📡 **SSE 实时流**：处理过程中逐条推送进度与结果，前端可实时更新
-  **单实例串行**：全局异步锁保证同一时间仅处理一个任务，资源可控
- 🔄 **双路输出**：结果同步写入本地 JSONL 文件 + 可选 HTTP 回调推送

---

## 🏗️ 数据处理流程

1. **请求接入** → `POST /stream_process`
2. **FastAPI 调度** → 请求锁控制 | SSE 流管理
3. **多模态流水线** → 视频/图片/音频 → 模型摘要 → 向量嵌入
4. **Milvus 存储** → Collection 管理 | HNSW 索引 | 分区存储
5. **结果分发** → 本地 JSONL 归档 | 外部 HTTP 回调 | SSE 实时推送

---

##  快速开始

### 📦 环境要求
- Python 3.11+
- Docker & Docker Compose（推荐生产部署）
- Milvus 2.6.6+
- 依赖服务：
  - `multimodal_service` (Qwen3.5-9B)：视频/图片摘要
  - `bgm_service` (Qwen-Audio-Chat)：音频摘要
  - `embed_service` (BGE-Large)：文本向量化

### 💻 本地开发
```bash
# 1. 克隆并进入项目
git clone git clone http://gogs.km360.cn/lituo/process-media.git
cd process-media

# 2. 安装依赖
uv sync

# 3. 修改配置
# 编辑 config/path_config.py 与 config/constant_config.py 填写各服务地址与路径

# 4. 启动服务
uvicorn video_director:app --host 0.0.0.0 --port 8013 --reload
```