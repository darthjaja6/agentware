from agentware.agent_logger import Logger
from agentware.memory import Memory
from agentware.base import BaseAgent, OneshotAgent, Connector
from agentware.hub import register_agent
import yaml
import openai
import re
import json
import os
import datetime
import copy

import traceback
from typing import List, Any, Dict
import agentware


logger = Logger()


class Agent(BaseAgent):

    @classmethod
    def fetch(cls, agent_id: int):
        logger.info("Creating agent from connector")
        agent = cls()
        memory = Memory.fetch(agent_id)
        agent.memory = memory
        agent.set_config(memory.get_main_agent_config())
        return agent

    def __init__(self):
        """ Creates a new agent instead of fetching from connector

        Construct the agent with or without memory.

        :param cfg: agent config used for initializing
        :type cfg: object

        :type arg2: The data type of the second argument
        :ivar attribute1: A description of the first attribute
        :ivar attribute2: A description of the second attribute
        """
        super().__init__()
        self.id = None
        self._memory = Memory()
        openai.api_key = agentware.openai_api_key

    def set_config(self, config: Dict):
        self._memory.set_main_agent_config(config)

    def get_config(self):
        return self._memory.get_main_agent_config()

    def run(self, prompt):
        output_valid = False
        num_retries = 0
        raw_output = ""
        format_instruction = f"{self.format_instruction}. {self.output_schema}"
        self._memory.prepare_run(self.prompt_prefix, prompt)
        parsed_output = ""
        logger.info(f"Making a working copy of memory: {self._memory}")
        memory_with_error = copy.deepcopy(self._memory)
        # Add format instruction to the last memory unit
        memory_with_error.delete_memory(-1)
        memory_with_error.add_memory({
            "role": self._memory.get_memory()[-1].role,
            "content": self._memory.get_memory()[-1].content + format_instruction
        })
        while True:
            if num_retries > 0:
                logger.debug(f"Retrying for the {num_retries} time")
            try:
                messages = memory_with_error.to_messages()
                logger.info(
                    f"memory before running is {self._memory}")
                raw_output = self._run(messages)
                try:
                    parsed_output = self.parse_output(raw_output)
                    new_memory = ""
                    if self.termination_observation in parsed_output:
                        new_memory = parsed_output[self.termination_observation]
                    self._memory.add_memory({
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

    def register(self, agent_id: str):
        assert agent_id
        register_agent(agent_id)
        self.agent_id = agent_id
        self._memory.agent_id = agent_id

    def push(self):
        # Check agent id valid
        assert self.agent_id
        logger.debug(f"Pushing agent with name {self.agent_id}")
        self._memory.update_check_point()
