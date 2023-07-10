import json

from agentware.base import OneshotAgent
from agentware.agent_logger import Logger

logger = Logger()


def TestReflectionQuestionAgent():
    cfg_path = ".configsreflection_question.json"
    cfg = None
    with open(cfg_path, "r") as f:
        cfg = json.loads(f.read())
    agent = OneshotAgent(cfg)
    logger.debug("agent created")
    logger.debug(
        agent.run("John Snow is one of the many main characters of Game of Thrones"))
