import unittest
from pymilvus import utility
import time
from agentware.base import Connector, Knowledge, BaseMilvusStore
from utils import DbClient, FakeCoreEngine

VECTOR_DB_CFG = {
    "uri": "130.211.121.33",
    "port": "19530",
    "user": "admin",
    "secure": "True",
    "nprobe": "32",
    "nlist": "1024",
    "collection_name": "test",
    "limit_distance": "1"
}


class ConnectorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.connector = Connector()
        cls.db_client = DbClient()
        cls.vector_db_client = BaseMilvusStore(VECTOR_DB_CFG)
        cls.core_engine = FakeCoreEngine()

    def setUp(self):
        # Clear database and vector db
        self.db_client.client.flushall()
        collections = utility.list_collections()
        for c in collections:
            utility.drop_collection(c)

    def test_search_unknown_knowledge(self):
        agent_id = 1
        new_knowledges = self.connector.search_knowledge(
            agent_id, self.core_engine.get_embeds("some new words"), 100)
        assert new_knowledges == []

    def test_save_remove_search_knowledge(self):
        agent_id = 1
        knowledge_texts = self.core_engine.get_sentences()
        knowledges = []
        for k in knowledge_texts:
            time.sleep(1)
            knowledges.append(Knowledge(
                time.time(), k, self.core_engine.get_embeds(k)))
        self.connector.save_knowledge(agent_id, knowledges)
        search_embeds = self.core_engine.get_embeds(
            "Microsoft")
        search_results = self.connector.search_knowledge(
            agent_id, search_embeds, 20)
        print('search results are', [k.content for k in search_results])
        assert [k.content for k in search_results] == ["Microsoft",
                                                       "Microsoft develops software.", "Apple is a tech giant.", "Bill Gates worked with Paul Allen."]
        recent_records = self.connector.get_recent_knowledge(
            agent_id, 20)
        assert [k.content for k in recent_records] == ["Microsoft", "Apple is a tech giant.",
                                                       "The sun shines brightly today.", "Google's search engine is powerful."]
        ids_to_remove = [r.id for r in recent_records[:2]]
        self.connector.remove_knowledge(agent_id, ids_to_remove)
        recent_records = self.connector.get_recent_knowledge(
            agent_id, 20)
        assert [k.content for k in recent_records] == ["The sun shines brightly today.",
                                                       "Google's search engine is powerful.", "Cats are adorable pets."]

    # def test_search_existing_knowledge(self):
    #     agent_id = 1

        # def test_save_knowledge(self):
        #     new_knowledges = self.connector.search_knowledge(
        #         self.agent_id, self._agent.get_embeds(keyword), token_limit=self.MAX_NUM_TOKENS_KNOWLEDGE)
        #     new_knowledges = self.connector.get_recent_knowledge(
        #         self.agent_id)
        #     self.connector.remove_knowledge(self.agent_id, ids_to_remove)
        #     self.connector.save_knowledge(self.agent_id, reflections)
        #     self.connector.update_longterm_memory(
        #         self.agent_id, memory_to_compress)
        #     self.connector.update_longterm_memory(
        #         self.agent_id, memory_to_compress)
        #     self.connector.update_checkpoint(
        #         self.agent_id,
        #         self._agent._config,
        #         self._memory,
        #         self._domain_knowledge,
        #         self._context)
        #     knowledge =
        #     self.connector
