"""
In this example, you will build your own agent
"""
import agentware

from agentware.hub import register_agent, push
from agentware.agent import Agent

if __name__ == '__main__':
    agentware.endpoint = "https://localhost:8741"
    agentware.openai_api_key = "your openai api key"
    agent = Agent()
    agent.register("personal assistant")
    agent.set_config({
        "description": "You are a personal assistant who knows the master's life details and can give assistance"
    })
    with agent.update():
        print(agent.run("I have a bottle of water on the second floor"))
        print(agent.run("Where can I get some water?"))
        print(agent.run("Then John took away the bottle"))
        print(agent.run("Where can I get something to drink?"))
