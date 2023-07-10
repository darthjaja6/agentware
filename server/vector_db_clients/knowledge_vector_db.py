from agentware.utils.num_token_utils import get_num_tokens
from typing import Dict, List
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from agentware.base import BaseMilvusStore, Knowledge
from agentware.agent_logger import Logger

logger = Logger()


class KnowledgeVectorStore(BaseMilvusStore):
    def __init__(self) -> None:
        super().__init__()

    def __init__(self, cfg: Dict[str, any]) -> None:
        super().__init__(cfg)

    def _create_collection(self, collection_name: str) -> Collection:
        id_field = FieldSchema(name="id", dtype=DataType.INT64,
                               is_primary=True, description="auto id")
        content_field = FieldSchema(
            name="content", dtype=DataType.VARCHAR, max_length=65535, description="content of knowledge, etc.")
        created_at_field = FieldSchema(
            name="created_at", dtype=DataType.INT64,  description="unix seconds, publish time")
        vector_field = FieldSchema(
            name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.DEFAULT_EMBED_DIM)

        schema = CollectionSchema(fields=[id_field, content_field, created_at_field, vector_field],
                                  auto_id=True,
                                  description="Knowledge base that contians ")
        collection = Collection(
            name=collection_name, schema=schema)
        index_params = {"index_type": "IVF_FLAT",
                        "metric_type": "L2", "params": {"nlist": self.nlist}}
        collection.create_index(
            field_name=vector_field.name, index_params=index_params)
        return collection

    def insert_knowledge(self, collection_name, knowledge_list: List[Knowledge]):
        self.check_and_maybe_create_collection(collection_name)
        print(
            f"Inserting knowledges {[knowledge.content for knowledge in knowledge_list]}")
        if not knowledge_list:
            return
        try:
            content = [knowledge.content for knowledge in knowledge_list]
            embeds = [knowledge.embeds for knowledge in knowledge_list]
            created_at = [knowledge.created_at for knowledge in knowledge_list]
            entities = [content, created_at, embeds]
            c = Collection(collection_name)
            ins_resp = c.insert(entities)
            if ins_resp:
                logger.debug(
                    f"Saved knowledge to collection {collection_name}")
        except Exception as err:
            print("get err {}".format(err))
            return err

    def query_by_url(self, url: str, collection_name: str) -> bool:
        self.check_and_maybe_create_collection(collection_name)
        c = Collection(collection_name)
        # query_embedding = None
        # search_params = {“nprobe”: 16}
        results = c.query(
            expr='url == "{}"'.format(url),
            output_fields=["id", "url", "content", "published_at", "vector"])
        if results:
            return results[0]
        return None

    def search_knowledge(self,  collection_name: str, query_embeds: List[float], token_limit: int):
        self.check_and_maybe_create_collection(collection_name)
        results = self._similarity_query(
            query_embeds, collection_name, ["content", "created_at"])
        retrived_knowledges = []
        total_num_tokens = 0
        for result in results[0]:
            content = result.entity.get('content')
            created_at = result.entity.get('created_at')
            num_tokens = get_num_tokens(content)
            total_num_tokens += num_tokens
            if total_num_tokens > token_limit:
                break
            retrived_knowledges.append(
                {"created_at": created_at, "content": content})
        return retrived_knowledges

    def query_after_publish_time(self, since_time: int, collection_name: str) -> List[any]:
        self.check_and_maybe_create_collection(collection_name)
        c = Collection(collection_name)
        # query_embedding = None
        # search_params = {“nprobe”: 16}
        results = c.query(
            expr='published_at > {}'.format(since_time),
            output_fields=["id", "url", "content", "published_at"])
        if results:
            return results
        return []

    def get_recent_knowledge(self, collection_name: str, token_limit) -> List[any]:
        self.check_and_maybe_create_collection(collection_name)
        # TODO: Implement
        return []
