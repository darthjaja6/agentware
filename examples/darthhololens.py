"""
In this example, you can fetch a public agent, darthhololens, and ask it questions.
"""
import agentware

from agentware.agent import Agent

if __name__ == '__main__':
    agentware.endpoint = "https://agentonia/api"
    agentware.openai_api_key = "your api key"
    agent = Agent.fetch("darthjaja/darthhololens")
    print(agent.run("Who are you?"))
    print(agent.run("Who is Darth Mickey"))
