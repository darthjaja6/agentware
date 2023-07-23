import openai

from langchain.chains import LLMChain
from langchain.agents.agent import AgentExecutor
from typing import Union, Dict, List


class CoreEngineBase():
    def run(self, messages: List[Dict[str, str]]):
        raise BaseException("Not implemented")


class OpenAICoreEngine(CoreEngineBase):
    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__()
        self.model = model

    def run(self, messages: List[Dict[str, str]]):
        return openai.ChatCompletion.create(
            model=self.model, messages=messages)


class LangchainCoreEngine(CoreEngineBase):
    def __init__(self, chain: Union[LLMChain, AgentExecutor]):
        super().__init__()
        self._chain = chain

    def run(self, messages: List[Dict[str, str]]):
        return self._chain.run(messages)
