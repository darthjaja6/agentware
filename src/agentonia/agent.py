import yaml
import openai
import re
import json
import os
import datetime
import copy

import traceback
from typing import List, Any, Dict
from agentware import openai_api_key
from agentware.base import BaseAgent, BaseConnector
from agentware.memory import Memory
from agentware.connector import Connector
from agentware.agent_logger import Logger

logger = Logger()

"""
一个agent在执行中可能涉及其它很多agent. agent大体分为两类,一类是全功能agent, 自带memory的。这种agent往往是
执行过程中动态创建, 用来执行一个新的任务。另一类是固定功能的agent, 并不自带memory. 第二类是否需要做存储? 需要的。
非全功能agent是否有context, knowledge, memory? 非全功能的agent需要有config, 这也包括context了。

agent_id对应一个agent的线上实体, 应该作为参数传入各种函数中, 而不应该定死。connect是否需要? 不再需要了。__init__
的时候做authentication, 然后动态维护jwt即可.

memory是属于每一个session的, knowledge不是. knowledge和commands应该是能被各种session access的. 应该有一个vector
db 用来做各种knowledge和command的搜索

1. 每个agent或者从纯本地文件中创建, 或者从connctor创建. 从connector创建要带agent_id.
2. session中带有子session的id, 用于查询。 不论哪种agent, 都开session. 要递归地拿session, 直到没有子session.
如何保证无环? 动态检测,每次创建session的时候确认不形成环
3.
"""


class Agent(BaseAgent):

    @classmethod
    def from_connector(cls, agent_id: int, connector: BaseConnector):
        """ Creates an agent from a session

        :param connector: A connector that connects to a memory
        :type arg1: The data type of the first argument
        :param arg2: A description of the second argument
        :type arg2: The data type of the second argument
        :ivar attribute1: A description of the first attribute
        :ivar attribute2: A description of the second attribute
        """
        logger.info("Creating agent from connector")

        memory = Memory.from_connector(agent_id, connector)
        return cls(agent_id, memory, connector)

    def __init__(self, memory: Memory, connector: BaseConnector, agent_id: int):
        """ Creates a new agent instead of fetching from connector

        Construct the agent with or without memory.

        :param cfg: agent config used for initializing
        :type cfg: object

        :type arg2: The data type of the second argument
        :ivar attribute1: A description of the first attribute
        :ivar attribute2: A description of the second attribute
        """
        self.connector = connector
        self.connector.get_token()
        if not agent_id:
            agent_id = self.connector.create_agent()
        logger.info(f"Created agent {agent_id}")
        self._agent_id = agent_id
        super().__init__(memory.get_main_agent_config())
        if not memory:
            raise ValueError("Memory must be valid")
        self.memory = memory
        openai.api_key = openai_api_key

    def run(self, prompt):
        output_valid = False
        num_retries = 0
        raw_output = ""
        format_instruction = f"{self.format_instruction}. {self.output_schema}"
        self.memory.prepare_run(self.prompt_prefix, prompt)
        parsed_output = ""
        logger.info(f"Making a working copy of memory: {self.memory}")
        memory_with_error = copy.deepcopy(self.memory)
        # Add format instruction to the last memory unit
        memory_with_error.delete_memory(-1)
        memory_with_error.add_memory({
            "role": self.memory.get_memory()[-1].role,
            "content": self.memory.get_memory()[-1].content + format_instruction
        })
        while True:
            if num_retries > 0:
                logger.debug(f"Retrying for the {num_retries} time")
            try:
                messages = memory_with_error.to_messages()
                logger.info(
                    f"memory before running is {self.memory}")
                raw_output = self._run(messages)
                try:
                    parsed_output = self.parse_output(raw_output)
                    new_memory = ""
                    if self.termination_observation in parsed_output:
                        new_memory = parsed_output[self.termination_observation]
                    self.memory.add_memory({
                        "role": "assistant",
                        "content": new_memory
                    })

                    output_valid = True
                    return parsed_output
                except Exception as err:
                    logger.warning(f"Error parsing with exception {err}")

                    traceback.print_exc()
                    if num_retries >= 1:
                        # Delete the previous error information
                        memory_with_error.delete_memory(-1)
                        memory_with_error.delete_memory(-1)
                    memory_with_error.add_memory({
                        "role": "assistant",
                        "content": raw_output
                    })
                    memory_with_error.add_memory({
                        "role": "user",
                        "content": f"Failed to parse output. Ignore all the format instructions you were given previously. Your output must be a json that strictly follow the schema while not including it {self.output_schema}"
                    })
            except Exception as e:
                logger.warning(f"Error getting agent output with error {e}")
            if output_valid or num_retries >= self.MAX_NUM_RETRIES:
                break
            num_retries += 1
        return parsed_output
