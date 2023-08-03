from agentware.memory import Memory
from agentware.base import create_helper_agents
import time
import json
import os


def create_memory_from_local_agents(main_agent_config_path, connector):
    base_agents = create_helper_agents()
    main_agent_config = None
    with open(main_agent_config_path, "r") as f:
        main_agent_config = json.loads(f.read())

    if "conversation_setup" in main_agent_config:
        context = main_agent_config["conversation_setup"]
    domain_knowledge = []
    memory_data = []
    memory = Memory(main_agent_config, base_agents, context,
                    domain_knowledge, memory_data, connector)
    return memory
