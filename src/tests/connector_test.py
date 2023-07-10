import time

from agentware.connector import Connector
from agentware.base import Knowledge


def TestConnector():
    # openai.Embedding.create(input=[text], model=cfg.get(cfg_section, "embed_model_name"))[
    #     'data'][0]['embedding']

    knowledges = [
        Knowledge(time.time())
    ]
    connector = Connector(1)
    connector.connect(agent_id="some_agent_id")
    # client.query_by_url(
    #     "https://news.bitcoin.com/okx-wallet-is-first-in-web3-to-utilize-leading-edge-mpc-technology-together-with-support-of-37-blockchains/")
