import hashlib
import json
import time
import asyncio
from typing import Literal
import requests
from pymilvus import AsyncMilvusClient, MilvusException, DataType, FieldSchema, CollectionSchema
from pymilvus.milvus_client import IndexParams
from requests.exceptions import HTTPError, RequestException
from config.constant_config import EMBEDDING_BATCH_SIZE, MATERIAL_TYPE, MATERIAL_CN, MATERIAL_EMOJI
from config.path_config import MILVUS_URL, MULTIMODEL_SERVICE_URL, BGM_SERVICE_URL, EMBED_SERVICE_URL, OUTPUT_DATA_PATH, \
    INPUT_DATA_PATH
from functionals.logger import process_media_logger
from functionals.stream_manager import StreamManager
from functionals.utils import user_id_to_collection_name, ModelServiceOrchestrator, ModelService, stream_post_payload

class LaunchMilvusAsync:
    def __init__(self, org: dict = {}, vector_db_url: str = MILVUS_URL, stream: StreamManager = None):
        # Create Milvus client
        self.client = AsyncMilvusClient(uri = vector_db_url, secure=False)

        # Streaming
        self.stream = stream

        # Local model switch (For RTX3090)
        self.MSO = ModelServiceOrchestrator()

        # Get info from data
        self.org_id:int = org["org_id"]
        self.material_sources = {
            "footage_regular": org.get("footage_regular", []),
            "footage_opening": org.get("footage_opening", []),
            "image": org.get("image", []),
            "bgm": org.get("bgm", [])
        }
        self.version: str = org.get("version", "0.0")

        # Async concurrency setting
        self.semaphore = asyncio.Semaphore(10) # concurrency control primitive that limits how many async tasks can run a specific block of code at the same time.

        # Create collection info
        self.collection_name = user_id_to_collection_name(self.org_id)
        self.limit = 10000 # limit for collection client query
        self.dimension = 1024 # dimension from the embedding model: BGE, Qwen0.6B - 1024; Qwen4B - 2560

    @staticmethod
    def _generate_unique_id(path: str) -> int:
        """Generate a deterministic ID for a path of the media."""
        hash_str = hashlib.sha256(f"{path}".encode()).hexdigest()
        # Convert to int and mask to 63 bits (safe for INT64)
        return int(hash_str[:16], 16) & ((1 << 63) - 1)

    @staticmethod
    async def _async_post(url: str, **kwargs):
        return await asyncio.to_thread(lambda: requests.post(url, **kwargs))

    async def _create_collection(self):
        """Create the Milvus collection schema and the partitions."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False,description="Unique deterministic ID from material_path hash"),
            FieldSchema(name="material_id", dtype=DataType.INT64,description="System-wide material identifier (non-unique)"),
            FieldSchema(name="material_path", dtype=DataType.VARCHAR, max_length=1024,description="Unique file path; changes when material is edited"),
            FieldSchema(name="industry_id", dtype=DataType.INT64,description="Tag-style metadata for industry classification"),
            FieldSchema(name="status", dtype=DataType.INT64, description="Active status: 0=inactive, 1=active"),
            FieldSchema(name="desc_json", dtype=DataType.JSON, description="Full multimodal LLM output as JSON"),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension,description="Embedding vector for semantic search"),
            FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=256, description="Version of the data, each update is a new version, even for entities without changes")
        ]
        schema = CollectionSchema(fields=fields,description=f"Multimodal media collection for org_id={self.org_id}")
        await self.client.create_collection(
            collection_name=self.collection_name,
            dimension = self.dimension,
            schema=schema
        )
        for partition_name in MATERIAL_TYPE:
            await self.client.create_partition(
                collection_name=self.collection_name,
                partition_name = partition_name
            )
        process_media_logger.info(f"已创建向包括partition的量数据库collection: {self.collection_name}")

    async def _create_hnsw_index(self):
        """Create HNSW index - called only once when collection is created."""
        try:
            # First, drop any existing indexes on the vector field
            existing_indexes = await self.client.list_indexes(self.collection_name)
            for index_name in existing_indexes:
                try:
                    await self.client.release_collection(self.collection_name)
                    await self.client.drop_index(self.collection_name, index_name)
                    process_media_logger.info(f"向量数据库collection: {self.collection_name}已删除现有index：{index_name}")
                except Exception as e:
                    process_media_logger.warning(f"向量数据库collection: {self.collection_name}删除index {index_name}时发生警告：{str(e)}")
            # Create HNSW index
            index_params = IndexParams()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200}
            )
            await self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            process_media_logger.info(f"已创建向量数据库collection: {self.collection_name}的HNSW index")
        except MilvusException as e:
            process_media_logger.info(f"创建向量数据库collection: {self.collection_name} HNSW index时发生错误：{str(e)}")
            raise RuntimeError(str(e))

    async def _ensure_hnsw_index(self):
        """确保HNSW索引存在, 不存在则创建"""
        try:
            await self.client.release_collection(self.collection_name)
            existing_indexes = await self.client.list_indexes(self.collection_name)

            hnsw_exists = False
            for index_name in existing_indexes:
                try:
                    index_info = await self.client.describe_index(self.collection_name, index_name)
                    if (index_info.get('index_type') == 'HNSW' and index_info.get('metric_type') == 'COSINE'):
                        hnsw_exists = True
                        process_media_logger.info(f"向量数据库collection: {self.collection_name} HNSW index已存在, 索引名称为{index_name}")
                        break
                except Exception as e:
                    process_media_logger.warning(f"向量数据库collection: {self.collection_name}检查索引{index_name}时发生警告：{str(e)}")

            if not hnsw_exists:
                process_media_logger.info(f"向量数据库collection: {self.collection_name}现有索引不是HNSW类型, 替换为HNSW")
                await self._create_hnsw_index()

        except Exception as e:
            process_media_logger.error(f"向量数据库collection: {self.collection_name}检查HNSW索引时发生错误：{str(e)}")
            raise RuntimeError(str(e))

    async def _get_existing_ids_with_retry(
            self,
            material_type: Literal[*MATERIAL_TYPE],
            max_retries: int = 3) -> set[int]:
        """get existing ID"""
        for attempt in range(max_retries):
            try:
                return await self._get_existing_ids(material_type)
            except Exception as e:
                process_media_logger.warning(
                    f"⚠️向量数据库collection: {self.collection_name}, partition: {material_type}"
                    f"获取现有ID失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                else:
                    process_media_logger.error(
                        f"❌ 向量数据库collection: {self.collection_name}, partition: {material_type}获取现有ID最终失败: {e}"
                    )
                    raise
        return set()

    async def _get_existing_ids(self, material_type: Literal[*MATERIAL_TYPE]) -> set[int]:
        """Fetch all existing phrase_id values from Milvus (with pagination)."""
        existing_ids = set()
        try:
            offset = 0
            while True:
                results = await self.client.query(
                    collection_name=self.collection_name,
                    filter="",
                    output_fields=["id"],
                    partition_names= [material_type],
                    limit=self.limit,
                    offset=offset
                )
                if not results:
                    break
                for r in results:
                    pid = r.get("id")
                    if pid is not None and pid != -1:  # exclude metadata sentinel
                        existing_ids.add(int(pid))
                if len(results) < self.limit:
                    break
                offset += self.limit
            return existing_ids
        except Exception as e:
            process_media_logger.warning(
                f"⚠️ 无法获取向量数据库collection: {self.collection_name}, partition: {material_type} 现有的ID列表：{e}"
            )
            return set()

    async def _prepare_target_data(self, data: list[dict]) -> tuple[set[int], list[dict], dict[int, dict]]:
        """Prepare target data: compute IDs and build mapping."""
        target_ids = set()
        target_ids_others = []
        id_to_data = {}
        # Iterate the data
        for item in data:
            # Validate the data
            if (isinstance(item.get("material_id"), int)
                    and isinstance(item.get("material_path"), str)
                    and isinstance(item.get("industry_id"), int)
                    and item.get("status") in {0, 1}):
                _id = self._generate_unique_id(item["material_path"])
                target_ids.add(_id)
                id_to_data[_id] = {**item, "id": _id}
                # item: id, material_id, material_path, industry_id, status
            else:
                target_ids_others.append(item)
                payload = {
                    "org_id": self.org_id,
                    "material_id": item.get("material_id"),
                    "material_path": item.get("material_path"),
                    "desc_json": None,
                    "version": self.version,
                }
                # output and request the unverified data now
                await stream_post_payload(payload, self.stream)

        return target_ids, target_ids_others, id_to_data

    async def _incremental_sync_data(self):
        """Incrementally sync data: insert new, delete obsolete."""
        process_media_logger.info(f"🔁 开始增量同步数据 - 组织ID: {self.org_id}, 量数据库collection: {self.collection_name}, 数据版本: {self.version}")
        for material_type, material_list in self.material_sources.items():
            try:
                process_media_logger.info(f"{MATERIAL_EMOJI[material_type]} 更新{MATERIAL_CN[material_type]}, 数据版本: {self.version}")
                target_ids, target_ids_others, id_to_data = await self._prepare_target_data(material_list)
                process_media_logger.info(
                    f"目标数据包含{MATERIAL_CN[material_type]}{len(material_list)}条, 验证失败{len(target_ids_others)}条, "
                    f"验证成功并将处理{len(target_ids)}条. 数据版本: {self.version}"
                )
                # id_to_data: {id: {id, material_id, material_path, industry_id, status}}

                existing_ids = await self._get_existing_ids_with_retry(material_type)
                process_media_logger.info(f"向量数据库collection: {self.collection_name}现有数据包含{MATERIAL_CN[material_type]}{len(existing_ids)}条")

                to_delete = existing_ids - target_ids # In current system, a real deletion will only make statu 0->1, this is usually empty
                to_keep = existing_ids & target_ids
                to_insert = target_ids - existing_ids

                if to_delete:
                    process_media_logger.info(f"向量数据库collection: {self.collection_name}将删除{len(to_delete)}条过期{MATERIAL_CN[material_type]}")
                    await self._batch_delete_material(list(to_delete), material_type) # No need partition name here

                if to_keep:
                    process_media_logger.info(f"向量数据库collection: {self.collection_name}将更新{len(to_keep)}条过期{MATERIAL_CN[material_type]}")
                    await self._batch_update_material(to_keep, id_to_data, material_type) # No need partition name here

                if to_insert:
                    process_media_logger.info(f"向量数据库collection: {self.collection_name}将插入{len(to_insert)}条{MATERIAL_CN[material_type]}")
                    material_list_to_insert = [id_to_data[_id] for _id in to_insert]
                    await self._insert_material(material_type, material_list_to_insert)

                if not to_delete and not to_keep and not to_insert:
                    process_media_logger.info(f"向量数据库collection: {self.collection_name}{MATERIAL_CN[material_type]}已同步, 无需更新")

                self._log_sync_stats(MATERIAL_CN[material_type], len(existing_ids), len(to_delete), len(to_insert), len(target_ids))

            except Exception as e:
                process_media_logger.error(f"向量数据库collection: {self.collection_name}{MATERIAL_CN[material_type]}更新失败：{str(e)}")
                raise
        process_media_logger.info(f"🔁 增量同步数据完成 - 组织ID: {self.org_id}, 向量数据库collection: {self.collection_name}, 数据版本: {self.version}")

    async def _batch_delete_material(self, ids: list[int], material_type: str, batch_size: int = 1000):
        """Batch delete materials (fallback to chunked deletion on failure)."""
        if not ids:
            return
        try:
            if len(ids)<=batch_size: # Try in one go for small amount deletion
                await self.client.delete(collection_name=self.collection_name, partition_name=material_type, ids=ids)
                process_media_logger.info(f"向量数据库collection: {self.collection_name}一次性删除{len(ids)}条{MATERIAL_CN[material_type]}内容")
            else: # Delete in batches for large amount
                for i in range(0, len(ids), batch_size):
                    batch = ids[i:i + batch_size]
                    await self.client.delete(collection_name=self.collection_name, partition_name=material_type, ids=batch)
                    process_media_logger.info(f"向量数据库collection: {self.collection_name}正在分批删除{MATERIAL_CN[material_type]}, 已删除批次 {i // batch_size + 1}: {len(batch)} 条")
                    await asyncio.sleep(0.01)  # Delay a bit to reduce server pressure
                process_media_logger.info(f"向量数据库collection: {self.collection_name}分批删除{MATERIAL_CN[material_type]}完成, 共{len(ids)}条")
        except Exception as e:
            process_media_logger.warning(f"向量数据库collection: {self.collection_name}删除{MATERIAL_CN[material_type]}失败, 尝试降级方案：{str(e)}")
            await self._delete_with_filter_fallback(ids, material_type)

    async def _delete_with_filter_fallback(self, ids: list[int], material_type: str):
        try:
            id_str = ", ".join(str(_id) for _id in ids)
            filter_expr = f"id in [{id_str}]"
            await self.client.delete(
                collection_name=self.collection_name,
                partition_name=material_type,
                filter=filter_expr
            )
        except Exception as e:
            process_media_logger.error(f"向量数据库collection: {self.collection_name} filter删除{MATERIAL_CN[material_type]}也失败, 尝试逐个删除：{str(e)}")
            # Delete one by one
            success_count = 0
            for _id in ids:
                try:
                    await self.client.delete(
                        collection_name=self.collection_name,
                        partition_name=material_type,
                        filter=f"id == {_id}"
                    )
                    success_count += 1
                except Exception as e:
                    process_media_logger.warning(f"向量数据库collection: {self.collection_name}无法删除{MATERIAL_CN[material_type]}ID {_id}: {e}")
            process_media_logger.info(f"向量数据库collection: {self.collection_name}逐个删除{MATERIAL_CN[material_type]}完成：{success_count}/{len(ids)}成功")

    async def _batch_update_material(self, ids: set[int], id_to_data:dict, material_type: str, batch_size: int = 1000):
        """Batch update materials."""
        upsert_data = [{**id_to_data[_id],"version":self.version} for _id in ids] # Remember to include version in the updated data
        if not ids:
            return
        try:
            if len(ids) <= batch_size:  # Try in one go for small amount deletion
                await self.client.upsert(
                    collection_name=self.collection_name,
                    partition_name=material_type,#important for partial update
                    data=upsert_data,
                    partial_update=True
                )
                process_media_logger.info(f"向量数据库collection: {self.collection_name}一次性更新{len(ids)}条{MATERIAL_CN[material_type]}")
            else:  # Delete in batches for large amount
                for i in range(0, len(ids), batch_size):
                    batch = upsert_data[i:i + batch_size]
                    await self.client.upsert(
                        collection_name=self.collection_name,
                        partition_name=material_type,
                        data=batch,
                        partial_update=True
                    )
                    process_media_logger.info(
                        f"向量数据库collection: {self.collection_name}正在分批更新{MATERIAL_CN[material_type]}, 已更新批次 {i // batch_size + 1}: {len(batch)} 条")
                    await asyncio.sleep(0.01)  # Delay a bit to reduce server pressure
                process_media_logger.info(f"向量数据库collection: {self.collection_name}分批更新{MATERIAL_CN[material_type]}完成, 共{len(ids)}条")
        except Exception as e:
            process_media_logger.warning(f"向量数据库collection: {self.collection_name}更新{MATERIAL_CN[material_type]}失败, 尝试降级方案：{str(e)}")
            await self._update_with_fallback(upsert_data, material_type)

    async def _update_with_fallback(self, upsert_data: list[dict], material_type: str):
        success_count = 0

        # update one by one
        for u_d in upsert_data:
            try:
                await self.client.upsert(
                    collection_name=self.collection_name,
                    partition_name=material_type,
                    data = u_d,
                    partial_update=True
                )
                success_count += 1
            except Exception as e:
                process_media_logger.warning(f"向量数据库collection: {self.collection_name}无法更新{MATERIAL_CN[material_type]}ID {u_d.get('id')}: {e}")

        process_media_logger.info(f"向量数据库collection: {self.collection_name}逐个更新{MATERIAL_CN[material_type]}完成：{success_count}/{len(upsert_data)}成功")

    async def _prepare_insert_batch_with_concurrency(self, ids: list[int], id_to_data: dict[int, dict], material_type:str, semaphore: asyncio.Semaphore) -> list[dict]:
        """Prepare a batch of data with summarization and embeddings."""
        tasks = []
        for _id in ids:
            data = id_to_data.get(_id)
            # data: id, material_id, material_path, industry_id, status
            if data:
                task = self._prepare_single_material(_id, data, material_type, semaphore)
                # task: id, material_id, material_path, industry_id, status, desc_json, vector, version
                tasks.append(task)

        # run concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        batch_data = []
        for result in results:
            if isinstance(result, dict):
                batch_data.append(result)
        #batch_data: [data: id, material_id, material_path, industry_id, status, desc_json, vector, version]
        return batch_data

    async def _summarize_footage(self, path:str) -> dict:
        # Request the summarization service
        self.MSO.switch_to(ModelService.MULTIMODAL_SERVICE)
        try:
            response = await self._async_post(  # ✅ Non-blocking
                f"{MULTIMODEL_SERVICE_URL}/summarize_footage",
                json={"footage_path": path},
                timeout=600
            )
            response.raise_for_status()  # Explicitly catches 4xx/5xx HTTP errors
            response = response.json()
            success = response.get("success")

            if not isinstance(success, bool):
                e_m = f"❌ 响应的success应为True或False, 实际为{type(success).__name__}"
                process_media_logger.error(e_m)
                raise TypeError(e_m)

            if success:
                desc_json = response.get("data")
                if not isinstance(desc_json, dict):
                    e_m = f"❌ '{path}'响应数据格式无效, 预期为 dict, 实际为 {type(desc_json).__name__}"
                    process_media_logger.error(e_m)
                    raise ValueError(e_m)
            else: #response not success
                e_m=f"❌ 响应返回错误: {response.get('error_code')} - {response.get('error')}"
                process_media_logger.error(e_m)
                raise ValueError(e_m)

        except HTTPError as e:
            e_m = f"❌ 总结'{path}'HTTP 错误 {e.response.status_code}: {e}"
            process_media_logger.error(e_m)
            raise RuntimeError(e_m) from e

        except RequestException as e:
            # Catches ConnectionError, Timeout, TooManyRedirects, etc.
            e_m = f"❌ 总结'{path}'网络请求异常: {type(e).__name__} - {e}"
            process_media_logger.error(e_m)
            raise ConnectionError(e_m) from e

        except Exception as e:
            e_m = f"❌ 总结'{path}'异常: {type(e).__name__} - {e}"
            process_media_logger.error(e_m)
            raise RuntimeError(e_m) from e

        # Integrate the description
        segments = desc_json.get("segments", [])
        overall_summary = desc_json.get("overall_summary", "")

        segment_desc = ""
        for i in range(len(segments)):
            segment_desc += f"\n- 第{i + 1}段：{segments[i].get('description', '无内容')}"

        desc_json["overall_summary"] = overall_summary + '分段描述：' + segment_desc

        return desc_json

    async def _summarize_image(self, path:str) -> dict:
        self.MSO.switch_to(ModelService.MULTIMODAL_SERVICE)
        try:
            response = await self._async_post(  # ✅ Non-blocking
                f"{MULTIMODEL_SERVICE_URL}/summarize_image",
                json={"image_path": path},
                timeout=600
            )
            response.raise_for_status()  # Explicitly catches 4xx/5xx HTTP errors
            response = response.json()
            success = response.get("success")

            if not isinstance(success, bool):
                e_m = f"❌ 响应的success应为True或False, 实际为{type(success).__name__}"
                process_media_logger.error(e_m)
                raise TypeError(e_m)

            if success:
                desc_json = response.get("data")
                if not isinstance(desc_json, dict):
                    e_m = f"❌ '{path}'响应数据格式无效, 预期为 dict, 实际为 {type(desc_json).__name__}"
                    process_media_logger.error(e_m)
                    raise ValueError(e_m)
            else:  # response not success
                e_m = f"❌ 响应返回错误: {response.get('error_code')} - {response.get('error')}"
                process_media_logger.error(e_m)
                raise ValueError(e_m)

        except HTTPError as e:
            e_m = f"❌ 总结'{path}'HTTP 错误 {e.response.status_code}: {e}"
            process_media_logger.error(e_m)
            raise RuntimeError(e_m) from e

        except RequestException as e:
            # Catches ConnectionError, Timeout, TooManyRedirects, etc.
            e_m = f"❌ 总结'{path}'网络请求异常: {type(e).__name__} - {e}"
            process_media_logger.error(e_m)
            raise ConnectionError(e_m) from e

        except Exception as e:
            e_m = f"❌ 总结'{path}'异常: {type(e).__name__} - {e}"
            process_media_logger.error(e_m)
            raise RuntimeError(e_m) from e

        return desc_json

    async def _summarize_bgm(self, path:str) -> dict:
        self.MSO.switch_to(ModelService.BGM_SERVICE)
        try:
            response = await self._async_post(  # ✅ Non-blocking
                f"{BGM_SERVICE_URL}/summarize_bgm",
                json={"bgm_path": path},
                timeout=600
            )
            response.raise_for_status()  # Explicitly catches 4xx/5xx HTTP errors
            response = response.json()
            success = response.get("success")

            if not isinstance(success, bool):
                e_m = f"❌ 响应的success应为True或False, 实际为{type(success).__name__}"
                process_media_logger.error(e_m)
                raise TypeError(e_m)

            if success:
                desc_json = response.get("data")
                if not isinstance(desc_json, dict):
                    e_m = f"❌ '{path}'响应数据格式无效, 预期为 dict, 实际为 {type(desc_json).__name__}"
                    process_media_logger.error(e_m)
                    raise ValueError(e_m)
            else:  # response not success
                e_m = f"❌ 响应返回错误: {response.get('error_code')} - {response.get('error')}"
                process_media_logger.error(e_m)
                raise ValueError(e_m)

        except HTTPError as e:
            e_m = f"❌ 总结'{path}'HTTP 错误 {e.response.status_code}: {e}"
            process_media_logger.error(e_m)
            raise RuntimeError(e_m) from e

        except RequestException as e:
            # Catches ConnectionError, Timeout, TooManyRedirects, etc.
            e_m = f"❌ 总结'{path}'网络请求异常: {type(e).__name__} - {e}"
            process_media_logger.error(e_m)
            raise ConnectionError(e_m) from e

        except Exception as e:
            e_m = f"❌ 总结'{path}'异常: {type(e).__name__} - {e}"
            process_media_logger.error(e_m)
            raise RuntimeError(e_m) from e

        return desc_json

    async def _embed_summary(self, overall_summary:str) -> list:
        try:
            response = await self._async_post(  # ✅ Non-blocking
                f"{EMBED_SERVICE_URL}/embed",
                json={"input": overall_summary}
            )
            response.raise_for_status()  # Explicitly catches 4xx/5xx HTTP errors
            vector = response.json()["embeddings"][0]

            if not isinstance(vector, list):
                e_m = f"❌ 嵌入'{overall_summary[:5]}...'向量格式无效, 预期为 list, 实际为 {type(vector).__name__}"
                process_media_logger.error(e_m)
                raise ValueError(e_m)

            if len(vector) != self.dimension:
                e_m = f"❌ '{overall_summary[:5]}...'嵌入向量维度错误, 预期为 {self.dimension}, 实际为 {len(vector)}"
                process_media_logger.error(e_m)
                raise ValueError(e_m)

        except HTTPError as e:
            e_m = f"❌ 嵌入'{overall_summary[:5]}...'HTTP 错误 {e.response.status_code}: {e}"
            process_media_logger.error(e_m)
            raise RuntimeError(e_m) from e

        except RequestException as e:
            # Catches ConnectionError, Timeout, TooManyRedirects, etc.
            e_m = f"❌ 嵌入'{overall_summary[:5]}...'网络请求异常: {type(e).__name__} - {e}"
            process_media_logger.error(e_m)
            raise ConnectionError(e_m) from e

        except Exception as e:
            e_m = f"❌ 嵌入'{overall_summary[:5]}...'未知异常: {type(e).__name__} - {e}"
            process_media_logger.error(e_m)
            raise RuntimeError(e_m) from e

        return vector

    async def _prepare_single_material(self, _id: int, data: dict, material_type: str, semaphore: asyncio.Semaphore) -> dict|None:
        async with semaphore:
            # Use multimodal modal to understand the image/video footage/bgm first
            # data: id, material_id, material_path, industry_id, status
            # Summarize the material
            try:
                if material_type in ("footage_regular", "footage_opening"):
                    data["desc_json"] = await self._summarize_footage(data["material_path"])
                elif material_type == "image":
                    data["desc_json"] = await self._summarize_image(data["material_path"])
                else: #material_type == "bgm"
                    data["desc_json"] = await self._summarize_bgm(data["material_path"])
            except Exception as e:
                e_m = f"❌ 总结{data.get('material_id')} - {data.get('material_path')}失败: {e}"
                process_media_logger.error(e_m)
                return None
            # data: id, material_id, material_path, industry_id, status, desc_json

            # Embed the summary
            overall_summary = data["desc_json"].get("overall_summary", "")
            try:
                data["vector"] = await self._embed_summary(overall_summary)
            except Exception as e:
                e_m = f"❌ 嵌入{data.get('material_id')} - {data.get('material_path')}总结失败: {e}"
                process_media_logger.error(e_m)
                return None
            # data: id, material_id, material_path, industry_id, status, desc_json, vector

            # Add version
            data["version"] = self.version
            # data: id, material_id, material_path, industry_id, status, desc_json, vector, version
            return data

    async def _insert_all_data(self):
        """Insert all data on first-time collection creation."""
        process_media_logger.info(f"🆕 开始首次处理并插入数据 - 组织ID: {self.org_id}, 向量数据库collection: {self.collection_name}, 数据版本: {self.version}")
        for material_type, material_list in self.material_sources.items():
            if material_list:
                await self._insert_material(material_type, material_list)
        process_media_logger.info(f"🆕 首次处理并插入数据完成 - 组织ID: {self.org_id}, 向量数据库collection: {self.collection_name}, 数据版本: {self.version}")

    async def _insert_material(self, material_type: str, material_list: list[dict]):
        process_media_logger.info(f"{MATERIAL_EMOJI[material_type]} 处理+插入新增{MATERIAL_CN[material_type]}, 数据版本: {self.version}")
        # Verify the data and generate id, output the unverified data
        material_list_verified = []
        material_list_others = [] # cannot add dict into a set
        for item in material_list:
            if (isinstance(item.get("material_id"), int)
                    and isinstance(item.get("material_path"), str)
                    and isinstance(item.get("industry_id"), int)
                    and item.get("status") in {0, 1}):
                material_list_verified.append({**item, "id": self._generate_unique_id(item["material_path"])})
                # item: id, material_id, material_path, industry_id, status
            else:
                material_list_others.append(item)
                # output the unverified data now
                payload = {
                    "org_id": self.org_id,
                    "material_id": item.get("material_id"),
                    "material_path": item.get("material_path"),
                    "desc_json": None,
                    "version": self.version,
                }
                await stream_post_payload(payload, self.stream)
        # process the verified data now
        len_material_list_verified = len(material_list_verified)
        process_media_logger.info(
            f"{MATERIAL_CN[material_type]}{len(material_list)}条, 验证失败{len(material_list_others)}条, "
            f"验证成功并将处理{len_material_list_verified}条. 数据版本: {self.version}"
        )
        # Prepare and insert the data in batch
        for i in range(0, len_material_list_verified, EMBEDDING_BATCH_SIZE):
            process_media_logger.info(
                f"{MATERIAL_CN[material_type]}本批条次({i + 1}-{min((i + EMBEDDING_BATCH_SIZE), len_material_list_verified)})/{len_material_list_verified}"
            )
            # Prepare the batch data
            batch = material_list_verified[i:i + EMBEDDING_BATCH_SIZE]
            data_map = {item["id"]: item for item in batch}
            # item: id, material_id, material_path, industry_id, status

            insert_batch = await self._prepare_insert_batch_with_concurrency(
                list(data_map.keys()),
                data_map,
                material_type,
                self.semaphore
            )
            # insert_batch: [data: id, material_id, material_path, industry_id, status, desc_json, vector, version]

            # Verify the data as there may be None element inside
            insert_batch_verified = [j for j in insert_batch if isinstance(j, dict)]
            if not insert_batch_verified:
                # Still output and request all the information from the batch
                for item in batch:
                    payload = {
                        "org_id": self.org_id,
                        "material_id": item.get("material_id"),
                        "material_path": item.get("material_path"),
                        "desc_json": None,
                        "version": self.version,
                    }
                    await stream_post_payload(payload, self.stream)
                process_media_logger.error(
                    f"处理{MATERIAL_CN[material_type]}, "
                    f"本批条次({i + 1}-{min((i + EMBEDDING_BATCH_SIZE), len_material_list_verified)})/{len_material_list_verified}"
                    f"全部没有准备成功, 数据版本: {self.version}. 开始准备下一批次.")
                continue
            process_media_logger.info(
                f"处理{MATERIAL_CN[material_type]}, "
                f"本批条次({i + 1}-{min((i + EMBEDDING_BATCH_SIZE), len_material_list_verified)})/{len_material_list_verified}"
                f"原有{len(insert_batch)}条, 准备成功{len(insert_batch_verified)}条, 数据版本: {self.version}."
            )
            # Insert the batch data to Milvus collection
            try:
                await self.client.insert(
                    collection_name=self.collection_name,
                    partition_name=material_type,
                    data=insert_batch_verified
                )
                # insert_batch_verified: [data: id, material_id, material_path, industry_id, status, desc_json, vector, version]

                # Output the result, include all the information from the batch
                # Use material path as the unique identifier as some material may not have id:
                material_path_ibv = [item.get("material_path") for item in insert_batch_verified]

                for item in batch:  # batch is the current slice of footage_regular_verified
                    mp = item.get("material_path")
                    if mp in material_path_ibv:
                        inserted_item = next((d for d in insert_batch_verified if d.get("material_path") == mp))
                        payload = {
                            "org_id": self.org_id,
                            "material_id": inserted_item.get("material_id"),
                            "material_path": inserted_item.get("material_path"),
                            "desc_json": inserted_item.get("desc_json"),
                            "version": self.version,
                        }
                        await stream_post_payload(payload, self.stream)
                    else:
                        payload = {
                            "org_id": self.org_id,
                            "material_id": item.get("material_id"),
                            "material_path": item.get("material_path"),
                            "desc_json": None,
                            "version": self.version,
                        }
                        await stream_post_payload(payload, self.stream)

                process_media_logger.info(
                    f"🎉 插入{MATERIAL_CN[material_type]}至向量数据库collection: {self.collection_name}成功, "
                    f"并将插入数据和未通过处理数据输出至{OUTPUT_DATA_PATH}, "
                    f"本批条次({i + 1}-{min((i + EMBEDDING_BATCH_SIZE), len_material_list_verified)})/{len_material_list_verified}, "
                    f"数据版本: {self.version}"
                )
            except Exception as e:
                # Still output all the information from the batch
                for item in batch:
                    payload = {
                        "org_id": self.org_id,
                        "material_id": item.get("material_id"),
                        "material_path": item.get("material_path"),
                        "desc_json": None,
                        "version": self.version,
                    }
                    await stream_post_payload(payload, self.stream)
                process_media_logger.error(
                    f"❌ 插入{MATERIAL_CN[material_type]}至向量数据库collection: {self.collection_name}失败, "
                    f"本批条次({i + 1}-{min((i + EMBEDDING_BATCH_SIZE), len_material_list_verified)})/{len_material_list_verified}, "
                    f"数据版本: {self.version}: {e}"
                )
                continue
            await asyncio.sleep(0)

    def _log_sync_stats(self, material_name: str, existing_count: int, deleted_count: int, inserted_count: int, final_count: int):
        """Log sync statistics."""
        process_media_logger.info(
            f"向量数据库collection: {self.collection_name}更新{material_name}数据同步统计:\n"
            f"  原有记录: {existing_count}\n"
            f"  删除记录: {deleted_count}\n"
            f"  新增记录: {inserted_count}\n"
            f"  最终记录: {final_count}\n"
            f"  变化率: {(deleted_count + inserted_count) / max(existing_count, 1) * 100:.1f}%\n"
        )

    async def ensure_collection_ready(self):
        """Ensure collection exists and is ready with data."""
        start_time = time.time()
        process_media_logger.info(f"▶️ 开始操作向量数据库collection: {self.collection_name}, 数据版本: {self.version}")

        # Get collection status
        try:
            has_collection = await self.client.has_collection(self.collection_name, timeout=20.0)  # AWAIT
        except asyncio.TimeoutError:
            process_media_logger.error(f"⏰ 向量数据库collection: {self.collection_name} 连接超时 (20秒)")
            return None
        except Exception as e:
            process_media_logger.error(f"❌ 无法查询向量数据库collection: {self.collection_name} 是否存在: {str(e)}")
            return None

        # If collection doesn't exist, create it.
        if not has_collection:
            process_media_logger.info(f"ℹ️ 向量数据库collection: {self.collection_name}不存在, 开始创建......")
            await self._create_collection()
            await self._create_hnsw_index()
            process_media_logger.info(f"✅ 已创建新的向量数据库collection: {self.collection_name}及其HNSW index")
            try:
                await self.client.load_collection(self.collection_name, timeout=30)
                process_media_logger.info(f"🟢 向量数据库collection: {self.collection_name}已加载到内存")
            except Exception as e:
                process_media_logger.warning(f"❌ 加载向量数据库collection: {self.collection_name}失败, 尝试继续：{str(e)}")

            # insert latest data
            await self._insert_all_data()
        # If exists, update it.
        else:
            await self._ensure_hnsw_index()
            process_media_logger.info(f"ℹ️ 向量数据库collection: {self.collection_name}已存在, 使用已有的HNSW index")
            # load collection before query
            try:
                await self.client.load_collection(self.collection_name, timeout=30)
                process_media_logger.info(f"🟢 向量数据库collection: {self.collection_name}已加载到内存")
            except Exception as e:
                process_media_logger.warning(f"❌ 向量数据库collection: {self.collection_name}失败, 尝试继续：{str(e)}")

            # update with latest data
            await self._incremental_sync_data()

        elapsed = time.time() - start_time
        process_media_logger.info(f"🏁 向量数据库collection: {self.collection_name}任务完成, 数据版本: {self.version}, 耗时：{elapsed:.2f}秒")

        return

async def initialize_milvus_async(
        org: dict,
        stream: StreamManager = None,
        vector_db_url: str = MILVUS_URL
):
    milvus_launcher = LaunchMilvusAsync(org, vector_db_url, stream=stream)
    await milvus_launcher.ensure_collection_ready()

# test
if __name__ == "__main__":
    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    test_org = input_data[0]
    asyncio.run(initialize_milvus_async(test_org))