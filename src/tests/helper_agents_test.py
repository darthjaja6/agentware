import unittest
import os
import json

from agentware.base import OneshotAgent
from agentware.agent_logger import Logger
from agentware import HELPER_AGENT_CONFIGS_DIR_NAME
from agentware.core_engines import CoreEngineBase
from utils import DbClient, FakeCoreEngine
logger = Logger()


def get_config(config_fname):
    config_dir_absolute_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agentware", HELPER_AGENT_CONFIGS_DIR_NAME)
    with open(f"{config_dir_absolute_path}/{config_fname}", "r") as f:
        config = json.loads(f.read())
    return config


class HelperAgentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_client = DbClient()
        cls.fake_core_engine = FakeCoreEngine()

    def setUp(self):
        self.db_client.client.flushall()

    def tearDown(self):
        pass

    def test_conflict_resolver_agent(self):
        config = get_config("conflict_resolver.json")
        agent = OneshotAgent(config)

        class ConflictResolverCoreEngine(CoreEngineBase):

            def run(self, prompt):
                return "{\"wrong_facts\": [2]}"
        agent.set_core_engine(ConflictResolverCoreEngine())
        result = agent.run(
            '{\"observations\": [\"Daenerys bought the dragon\"], \"facts\": [{\"id\": 2, \"fact\": \"Daenerys has nothing\"}, {\"id\": 3, \"fact\": \"John snow is playing basketball with a dragon\"}]}')
        assert result == "[2]"


if __name__ == '__main__':
    unittest.main()
