from typing import Dict, List, Tuple
from agentware.base import MemoryUnit, Knowledge, Connector, OneshotAgent, BaseAgent
from agentware.agent_logger import Logger

import copy
import time
import json

logger = Logger()


class Memory():
    HELPER_AGENT_CONFIGS_DIR = "agentware/base_agent_configs"
    MAX_NUM_TOKENS_CONTEXT = 1000
    MAX_NUM_TOKENS_MEMORY = 1000
    MAX_NUM_TOKENS_KNOWLEDGE = 200

    def __init__(self):
        self._helper_agents = self.create_helper_agents()
        self._reflections = []
        self._commands = []
        self._domain_knowledge = []
        self._memory = []
        self._context = []
        self._num_tokens_memory = 0
        self._num_tokens_context = 0
        self._num_tokens_domain_knowledge = 0
        self._main_agent_config = dict()
        self.agent_id = None
        self.connector = Connector()

    @classmethod
    def fetch(cls, agent_id: int):
        connector = Connector()
        try:
            agent_config, memory_data, domain_knowledge, context = connector.get_checkpoint(
                agent_id)

            memory = cls.init(agent_config, context,
                              domain_knowledge, memory_data)
        except Exception as e:
            raise e
        return memory

    @classmethod
    def init(cls,
             main_agent_config: Dict[any, any],
             context: str,
             work_experience: List[Knowledge],
             working_memory: List[MemoryUnit]):
        logger.debug("initializing memory")
        memory = cls()
        memory.set_main_agent_config(main_agent_config)
        memory._context = context
        memory._work_experience = work_experience
        memory._memory = working_memory
        return memory

    def create_helper_agents(self) -> Dict[str, OneshotAgent]:
        facts_agent_config_path = f"{self.HELPER_AGENT_CONFIGS_DIR}/facts.json"
        ref_q_agent_config_path = f"{self.HELPER_AGENT_CONFIGS_DIR}/reflection_question.json"
        ref_agent_config_path = f"{self.HELPER_AGENT_CONFIGS_DIR}/reflection.json"
        summarize_agent_config_path = f"{self.HELPER_AGENT_CONFIGS_DIR}/summarize.json"
        tool_query_agent_config_path = f"{self.HELPER_AGENT_CONFIGS_DIR}/tool_query.json"

        agents = dict()
        fact_agent_config = None
        with open(facts_agent_config_path, "r") as f:
            fact_agent_config = json.loads(f.read())
        agents["fact"] = OneshotAgent(fact_agent_config)
        # Summarize agent
        summarizer_agent_config = None
        with open(summarize_agent_config_path, "r") as f:
            summarizer_agent_config = json.loads(f.read())
        agents["summarizer"] = OneshotAgent(summarizer_agent_config)
        # Reflection question
        ref_q_agent_config = None
        with open(ref_q_agent_config_path, "r") as f:
            ref_q_agent_config = json.loads(f.read())
        agents["reflection_q"] = OneshotAgent(ref_q_agent_config)
        # Reflection
        ref_agent_config = None
        with open(ref_agent_config_path, "r") as f:
            ref_agent_config = json.loads(f.read())
        agents["reflection"] = OneshotAgent(ref_agent_config)
        # Tool query
        tool_query_agent_config = None
        with open(tool_query_agent_config_path, "r") as f:
            tool_query_agent_config = json.loads(f.read())
        agents["tool_query"] = OneshotAgent(tool_query_agent_config)
        return agents

    def get_main_agent_config(self) -> Dict[str, any]:
        return self._main_agent_config

    def set_main_agent_config(self, config: Dict[str, any]):
        self._main_agent_config = config

    def get_num_tokens_memory(self):
        return self._num_tokens_memory

    def get_context(self) -> str:
        return self._context

    def get_domain_knowledge(self) -> List[Knowledge]:
        return self._domain_knowledge

    def get_commands(self) -> List[str]:
        return self._commands

    def get_reflections(self) -> List[Knowledge]:
        return self._reflections

    def get_memory(self) -> List[MemoryUnit]:
        return self._memory

    def get_helper_agents(self):
        return self._helper_agents

    def get_helper_agent_configs(self):
        return {agent_name: agent.get_config()
                for agent_name, agent in self._helper_agents.items()}

    def reflect_original(self, memory_text):
        """
        A TBD class
        这是按照 https://arxiv.org/pdf/2304.03442.pdf 来做的实现。
        保留了原本的设计, 但可能不适用于数据处理工作。不同的场景下, memory内容
        不一样, 有的是一系列事件的串联, 有的是针对某个话题的思辨。不同的memory
        内容各有适用的reflection机制
        """
        # Step1: Ask questions
        questions = self._helper_agents["reflection_q"].run(
            f"```{memory_text}``` Given only the information above, what are the most insightful questions we can ask about the subjects in the statements?”")
        answers = []
        # Step2: Answer the questions.
        # TODO: Follow the paper: https://arxiv.org/pdf/2304.03442.pdf and do similarity retrieval from agentware.memory, local knowledge and remote knowledge.
        for q in questions:
            answer = self._helper_agents["reflection"].run(
                f"```{memory_text}``` {q}. Answer the question above within 10 tokens")
            answers.append(answer)
        logger.debug(answers)

    def get_query_term(self, seed) -> str:
        return self._helper_agents["tool_query"].run(seed)

    def prepare_run(self, prompt_prefix: str, prompt: str):
        logger.info(
            f"Preparing for running prompt {prompt}\n and prefix {prompt_prefix}")
        super().prepare_run(prompt_prefix, prompt)
        # Get relevant knowledge
        keyword = self.get_query_term(prompt)
        logger.debug(f"search keywords for knowledge retrieval: {keyword}")
        new_knowledges = []
        if keyword:
            new_knowledges = self.connector.search_knowledge(
                self.agent_id, keyword, token_limit=self.MAX_NUM_TOKENS_DOMAIN_KNOWLEDGE)
        else:
            logger.debug(
                "Keyword is empty, fetching most recent knowledge instead")
            new_knowledges = self.connector.get_recent_knowledge(
                self.agent_id)
            self.update_knowledge(new_knowledges)
        # query for commands
        # self._commands = self.connector.search_commands(keyword)
        print("domain knowledges are", self._domain_knowledge)
        # print("commands are", self._commands)

    def reflect(self, memory_text, context=None) -> List[Knowledge]:
        # context gives information on what job the agent is doing.
        logger.debug("Making reflection from")
        logger.debug(memory_text)
        # Make reflection on compressed memory
        # Step1: Extract facts.
        # TODO: 解决first name / last name -> full name的问题
        # 发现facts和reflection比较冗余。暂时只保留reflection了
        facts = self._helper_agents["fact"].run(
            f"```{memory_text}```, extract all facts and insights concisely from the text above and make sure each fact has clear meaning if viewed independently. Each fact should not exceed 10 tokens.")
        facts = [f["subject_concept"] + " " + f["fact"] for f in facts]
        logger.debug(facts)
        knowledges = [Knowledge(int(time.time()), fact) for fact in facts]
        return knowledges

    def _compress_memory(self, reflect=False) -> Tuple[List[MemoryUnit], List[Knowledge]]:
        """ Compress memory and maybe do reflection

        Reflection is implemented in a similar way as the Stanford paper
        """
        # Let LLM summarize the first half of the memory
        # context and domain knowledge should not be changed
        # compress half of the user <-> assistant interaction
        # locate the half memory point
        logger.info("memory before compressing is")
        logger.info(self.__str__())
        compress_until_index = 0
        current_num_tokens = 0
        for i, m in enumerate(self._memory):
            print("m is", m)
            current_num_tokens += m.num_tokens
            if current_num_tokens > self.num_tokens_memory/2:
                compress_until_index = i
                break
        num_tokens_not_compressed = self.num_tokens_memory - current_num_tokens
        print(
            f"From {len(self._memory)} memory units, compressing from 0 to {compress_until_index}")
        memory_to_compress = self._memory[:(compress_until_index+1)]
        # Format memory to text
        memory_text = ""
        for m in memory_to_compress:
            memory_text += f"{m.role}: {m.content}\n"
        compressed_memory_content = self._helper_agents["summarizer"].run(
            f"```{memory_text}```. Please make a summary of the conversation above in no more than 200 tokens. Respond with the summarization")
        compressed_memory = MemoryUnit(
            "user",  f"A summary of our past conversation: {compressed_memory_content}")
        self._memory = [compressed_memory] + \
            self._memory[compress_until_index:]
        self.num_tokens_memory = num_tokens_not_compressed + compressed_memory.num_tokens
        logger.info("memory after compressing is")
        logger.info(self.__str__())
        reflections = self.reflect(memory_text)
        # Save checkpoint and add to knowledge
        if self.connector:
            self.connector.save_knowledge(self.agent_id, reflections)
            self.connector.update_longterm_memory(
                self.agent_id, memory_to_compress)
            # Update current checkpoint
            self.update_check_point(self.agent_id)
        return memory_to_compress

    def update_check_point(self):
        self.connector.update_checkpoint(
            self.agent_id,
            self._main_agent_config,
            self._memory,
            self._domain_knowledge,
            self._context)

    def add_memory(self, memory: Dict[str, str]):
        new_memory = MemoryUnit(memory["role"], memory["content"])
        self._memory.append(new_memory)
        self._num_tokens_memory += new_memory.num_tokens
        if self._num_tokens_memory > self.MAX_NUM_TOKENS_MEMORY:
            self._compress_memory(reflect=True)

    def update_knowledge(self, knowledges: List[Knowledge]):
        for k in knowledges:
            self._num_tokens_domain_knowledge += k.num_tokens
            if self._num_tokens_domain_knowledge > self.MAX_NUM_TOKENS_KNOWLEDGE:
                break

    def delete_memory(self, memory_index: int):
        if memory_index >= len(self._memory):
            logger.debug(
                f"Deleting index {memory_index} out of range of 0-{len(self._memory - 1)}")
        self._num_tokens_memory -= self._memory[memory_index].num_tokens
        del self._memory[memory_index]

    def to_messages(self):
        domain_knowledge_str = "No domain knowledge obtained"
        if self._domain_knowledge:
            domain_knowledge_str = f"Your domain knowledge is between the triple apostrophes: ```{';'.join([k.content for k in self._domain_knowledge])}```"
        commands_str = "No command"
        # TODO: Maybe it shouldn't be pronounced as
        if self._commands:
            commands_str = f"""The external tools you can use is between the triple apostrophes```{
                ';'.join([c.to_prompt() for c in self._commands])
            }```"""
        commands_str = ""
        logger.debug(f"context is { self._context}")
        return [
            {
                "role": "system",
                "content": self._context
            },
            {
                "role": "system",
                "content": domain_knowledge_str
            },
            {
                "role": "system",
                "content": commands_str
            }
        ] + [
            {
                "role": memory_unit.role,
                "content": memory_unit.content
            }
            for memory_unit in self._memory
        ]

    def __str__(self) -> str:
        prefix = f'\n************* Memory({self._num_tokens_memory + self._num_tokens_context + self._num_tokens_domain_knowledge} tokens) *************\n'
        suffix = f'\n************* End of Memory *************\n'
        context_str = f'\n<context> [{self._num_tokens_context} tokens]: {self._context}\n'
        knowledge_str = f"\n<knowledge>: [{self._num_tokens_domain_knowledge}]\n" + "\n----------------------\n".join(
            [k.__str__() for k in self.get_domain_knowledge()])
        memory_str = f"<memory> [{self._num_tokens_memory} tokens]:\n" + "\n----------------------\n".join(
            [m.__str__() for m in self.get_memory()])
        return prefix + context_str + knowledge_str + memory_str + suffix

    def __repr__(self):
        return self.__str__()

    def __deepcopy__(self, memodict={}):
        #          main_agent_config: Dict[any, any],
        #  agents: Dict[str, BaseAgent],
        #  context: str,
        #  work_experience: List[Knowledge],
        #  memory: List[MemoryUnit],
        #  connector: BaseConnector):
        cpyobj = type(self)(self._main_agent_config, self._helper_agents, self._context,
                            self._domain_knowledge, self._memory, self.connector)  # shallow copy of whole object
        cpyobj._domain_knowledge = copy.deepcopy(
            self._domain_knowledge, memodict)
        cpyobj._commands = copy.deepcopy(
            self._commands, memodict)
        cpyobj._memory = copy.deepcopy(self._memory, memodict)
        cpyobj._helper_agents = {agent_name: copy.deepcopy(
            agent) for agent_name, agent in self._helper_agents.items()}
        return cpyobj
