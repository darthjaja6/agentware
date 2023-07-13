from agentware.base import Connector

connector = Connector()


def register_agent(agent_id: str):
    if connector.register_agent(agent_id):
        raise ValueError(f"Agent {agent_id} exists")


def list_agents():
    return connector.all_agents()
