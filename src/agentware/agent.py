from agentware.agent_logger import Logger
from agentware.memory import Memory
from agentware.base import BaseAgent
from agentware.hub import register_agent, remove_agent, agent_exists
import openai
import json
import copy

import traceback
import agentware


logger = Logger()


class Agent(BaseAgent):

    @classmethod
    def pull(cls, agent_id: int):
        logger.info("Creating agent from connector")
        agent = cls()
        memory = Memory.pull(agent_id, agent)
        agent.memory = memory
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
        self._memory = Memory(self)
        self._update_mode = False
        openai.api_key = agentware.openai_api_key

    def update(self):
        return self.UpdateContext(self)

    class UpdateContext:
        def __init__(self, obj):
            self.obj = obj

        def __enter__(self):
            self.obj._update_mode = True
            return self.obj

        def __exit__(self, exc_type, exc_val, exc_tb):
            # update knowledge
            self.obj._update_mode = False

    def run(self, prompt):
        output_valid = False
        num_retries = 0
        raw_output = ""
        logger.debug(f"Adding prompt to memory: {prompt}")
        self._memory.update_context(self._prompt_prefix, prompt)
        self._memory.add_memory({
            "role": "user",
            "content": f"{self._prompt_prefix} {prompt} {self._format_instruction}."
        })
        memory_with_error = copy.deepcopy(self._memory)
        logger.info(f"Copy of memory made: {memory_with_error}")
        while True:
            if num_retries > 0:
                logger.debug(f"Retrying for the {num_retries} time")
            try:
                messages = memory_with_error.to_messages()
                raw_output = self._run(messages)
                try:
                    output = ""
                    if self._output_schema:
                        output = self.parse_output(raw_output)
                        if self._termination_observation in output:
                            # Whatever sub structure in outupt is dumped to str
                            output = json.dumps(
                                output[self._termination_observation])
                    else:
                        output = raw_output
                    logger.debug(f"Adding response to memory: {output}")
                    self._memory.add_memory({
                        "role": "assistant",
                        "content": output
                    })

                    output_valid = True
                    return output
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
                        "content": f"Failed to parse output. Ignore all the format instructions you were given previously. Your output must be a json that strictly follow the schema while not including it {self._output_schema}"
                    })
            except Exception as e:
                logger.warning(f"Error getting agent output with error {e}")
            if output_valid or num_retries >= self.MAX_NUM_RETRIES:
                break
            num_retries += 1

    def exists(self):
        if not self.id:
            return False
        return agent_exists(self.id)

    def remove(self):
        assert self.id
        remove_agent(self.id)

    def register(self, agent_id: str = ""):
        if not agent_id:
            agent_id = self.id
        assert agent_id
        register_agent(agent_id)
        self.id = agent_id
        self._memory.agent_id = agent_id

    def push(self):
        # Check agent id valid
        assert self.id
        logger.debug(f"Pushing agent with name {self.id}")
        self._memory.update_agent()
