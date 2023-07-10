from typing import Dict, List, Tuple
from agentware.base import BaseMemory, MemoryUnit, Knowledge, BaseConnector, OneshotAgent, BaseAgent
from agentware.agent_logger import Logger

import copy
import time

logger = Logger()


class Memory(BaseMemory):
    MAX_NUM_TOKENS_CONTEXT = 1000
    MAX_NUM_TOKENS_MEMORY = 1000
    MAX_NUM_TOKENS_DOMAIN_KNOWLEDGE = 200

    @classmethod
    def from_connector(cls, agent_id: int, connector: BaseConnector):
        agent_config, helper_agents_configs, memory_data, domain_knowledge, context = connector.get_checkpoint(
            agent_id)
        agents = {agent_name: OneshotAgent(
            agent_config) for agent_name, agent_config in helper_agents_configs.items()}

        return cls(agent_config, agents, context,
                   domain_knowledge, memory_data, connector)

    def __init__(self,
                 main_agent_config: Dict[any, any],
                 agents: Dict[str, BaseAgent],
                 context: str,
                 work_experience: List[Knowledge],
                 memory: List[MemoryUnit],
                 connector: BaseConnector):
        logger.debug("initializing memory")
        super().__init__(main_agent_config, context, work_experience, memory)
        agent_names = ["fact", "summarizer",
                       "reflection_q", "reflection", "tool_query"]
        for agent_name in agent_names:
            if not agent_name in agents:
                raise ValueError(f"Missing agent {agent_name}")
        self._helper_agents = agents
        self.connector = connector
        # Creating a new agent as this memory is created
        self.agent_id = self.connector.create_agent()

    def get_helper_agents(self):
        return self._helper_agents

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
        """
        reflect的主要作用是从短时memory中提取出来重要的部分存储下来。
        第一步是提取出来在当前topic下价值比较高的facts和insights.
        第二步是把这些facts, insights以自然语言的方式存储到vector db中, 作为最底层的长期记忆
        这就是那种时不时浮现出来的“只言片语”
        思考:
        knowledge graph更应该是在此基础上的一种索引. 这个索引从facts中取得。knowledge graph
        相对于vector db有什么优势? 相对于LLM + embedding search, 可能没有任何优势
        step1: 解决entity的问题. 比如Mill vs John Staurt Mill. 这个可以以类似于kg entity的
        方式解决。让agent先返回一个entity列表, 然后规定只能往这个里面存东西
        """
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
            helper_agent_configs = {agent_name: agent.get_config()
                                    for agent_name, agent in self._helper_agents.items()}
            self.connector.update_checkpoint(
                self.agent_id,
                self._main_agent_config,
                helper_agent_configs,
                self._memory,
                self._domain_knowledge,
                self._context)
        return memory_to_compress, reflections

    def __str__(self) -> str:
        return super().__str__()

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
