import unittest

from agentware import hub
from agentware.base import OneshotAgent, Connector
from agentware.agent import Agent
from agentware.agent_logger import Logger
from utils import DbClient, FakeCoreEngine
logger = Logger()

TEST_CFG = {
    "name": "test agent",
            "conversation_setup": "You are DarthHololens, an AI assistant who lives in the universe of Star Wars series and knows everything about this world",
            "prompt_prefix": "Constraint: answer in no more than 200 tokens"
}


class AgentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_client = DbClient()
        cls.fake_core_engine = FakeCoreEngine()

    def setUp(self):
        self.db_client.client.flushall()

    def tearDown(self):
        pass

    def test_register_agent(self):
        agent_id = "some agent idz"
        hub.register_agent(agent_id)
        agent_ids = hub.list_agents()
        print('agents are', agent_ids)
        assert agent_ids[0] == agent_id

    def test_duplicate_register_fail(self):
        agent_id = "some_agent_id"
        hub.register_agent(agent_id)
        self.assertRaises(ValueError, hub.register_agent, agent_id)

    def test_fail_to_fetch_unexist(self):
        self.assertRaises(ValueError, Agent.pull, "some/unexisted")

    def test_register_and_push_and_fetch_and_run_agent(self):
        agent = Agent()
        agent_id = "some_agent_name"
        agent.register(agent_id)
        agent.set_config(TEST_CFG)
        agent.push()
        agents = hub.list_agents()
        print("agents are", agents)
        assert agents[0] == agent_id
        updated_agent = Agent.pull(agent_id)
        assert updated_agent.get_config() == TEST_CFG

    def test_memory_compression(self):
        agent = Agent.init(TEST_CFG)
        for i in range(10):
            agent.run("What is your name?")

    def test_agent_reflection_run_in_update_mode(self):
        agent = Agent.init(TEST_CFG)
        agent.set_core_engine(self.fake_core_engine)
        assert agent._update_mode == False
        with agent.update():
            assert agent._update_mode == True
            agent.run("What is your name?")
        assert agent._update_mode == False


if __name__ == '__main__':
    unittest.main()
