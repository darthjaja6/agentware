"""
This example shows several ways of creating/fetching agents:
1. Creating an agent from a local config file
2. Fetching an existing agent from cloud
"""
import json
import math
from agentware.agent import Agent
from agentware.agent_logger import Logger
from utils import create_memory_from_local_agents

logger = Logger()


def get_chunks(text: str, span_length: int):
    num_tokens = len(text)
    print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) /
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
    chunks = []
    for boundary in spans_boundaries:
        chunks.append({
            "text": text[boundary[0]:boundary[1]],
            "boundary": boundary
        })
    return chunks


def create_agent_from_local_config():
    agent_cfg_path = "./configs/document_reader.json"
    connector_cfg_path = "./configs/connector_config.json"
    connector_cfg = None
    with open(connector_cfg_path, "r") as f:
        connector_cfg = json.loads(f.read())
    connector = Connector(connector_cfg)
    memory = create_memory_from_local_agents(agent_cfg_path, connector)
    agent = Agent(memory, connector, None)

    # Get local text
    text = ""
    with open("capital_theory.txt", "r") as f:
        text = f.read()
    chunks = get_chunks(text, 400)
    for chunk in chunks:
        output = agent.run(
            f"Given the text, ```{chunk['text']}```, please summarize the text in the quotes above")
        print("response is ", output)


def create_agent_from_local_connector(agent_id):
    connector_cfg_path = "./configs/connector_config.json"
    connector_cfg = None
    with open(connector_cfg_path, "r") as f:
        connector_cfg = json.loads(f.read())
    connector = Connector(1, connector_cfg)
    agents = connector.all_agents(user_id=1)
    print("agents are", agents)
    connector.connect(agent_id)
    agent = Agent.from_connector(connector)
    # agent = Agent.from_connector(
    #     connector, selected_agent_id, agent_graph)
    # memory = create_memory(cfg_path, connector)
    # agent = Agent(cfg, memory, connector)
    # # Get document feed
    # links = get_news_links("Google", pages=1, max_links=3)
    # for url in links:
    #     try:
    #         print("getting article")
    #         article = get_article(url)
    #     except:
    #         print(f"failed to get article from {url}")
    #         continue
    #     chunks = get_chunks(article.text, 2000)
    #     config = {
    #         "article_title": article.title,
    #         "article_publish_date": article.publish_date
    #     }
    #     for chunk in chunks:
    #         output = agent.run(
    #             f"Given the text, ```{chunk['text']}```, please summarize the text in the quotes above")
    #         print("response is ", output)

    # Get local text
    text = ""
    with open("capital_theory.txt", "r") as f:
        text = f.read()
    chunks = get_chunks(text, 400)
    for chunk in chunks:
        output = agent.run(
            f"Given the text, ```{chunk['text']}```, please summarize the text in the quotes above")
        print("response is ", output)


if __name__ == "__main__":
    # create_agent_from_local_config()
    create_agent_from_connector()
