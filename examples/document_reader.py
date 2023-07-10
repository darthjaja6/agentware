"""
This example shows 
1. Creating an agent from a local config file
2. Fetching an existing agent from cloud
"""
from agentware.agent import Agent
from agentware.connector import Connector
from agentware.agent_logger import Logger

logger = Logger()

connector = Connector({
    # "endpoint": "https://demo.agentware.ai/api/v0"
    "endpoint": "http://localhost:8000"
})
print("Public agents are", connector.list_agents())
agent = Agent.from_connector(1, connector)
