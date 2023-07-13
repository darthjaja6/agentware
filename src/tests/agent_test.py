import unittest
import json
import agentware

from agentware import hub
from agentware.base import OneshotAgent, Connector
from agentware.agent import Agent
from agentware.agent_logger import Logger
from tests.utils import DbClient
logger = Logger()

TEST_CFG = {
    "name": "test/3PO",
    "description": ""
}


class AgentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_client = DbClient()

    def setUp(self):
        self.db_client = DbClient()

    def tearDown(self):
        self.db_client.client.flushall()

    def test_register_agent(self):
        agent_id = "some agent id"
        hub.register_agent(agent_id)
        agent_ids = hub.list_agents()
        assert agent_ids[0] == agent_id

    def test_duplicate_register_fail(self):
        agent_id = "some_agent_id"
        hub.register_agent(agent_id)
        self.assertRaises(ValueError, hub.register_agent, agent_id)

    def test_fail_to_fetch_unexist(self):
        self.assertRaises(ValueError, Agent.fetch, "some/unexisted")

    def test_register_and_push_and_fetch_and_run_agent(self):
        agent = Agent()
        agent_id = "some_agent_name"
        agent.register(agent_id)
        config = {
            "name": "test agent",
            "conversation_setup": "You are DarthHololens, an AI assistant who lives in the universe of Star Wars series and knows everything about this world",
            "prompt_prefix": "Constraint: answer in no more than 200 tokens"
        }
        agent.set_config(config)
        agent.push()
        agents = hub.list_agents()
        assert agents[0] == agent_id
        updated_agent = Agent.fetch(agent_id)
        assert updated_agent.get_config() == config
        # Run some queries until compression happens


if __name__ == '__main__':
    unittest.main()
