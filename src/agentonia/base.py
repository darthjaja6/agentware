import os
from typing import List, Dict, Tuple
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from pymilvus import Milvus, connections, utility
from agentware.agent_logger import Logger
import json
import openai
import copy

from datetime import datetime
from agentware.utils.num_token_utils import count_message_tokens
from typing import List, Dict
import time

from agentware.utils.json_validation.validate_json import validate_json
from agentware.utils.json_fixes.parsing import fix_and_parse_json

logger = Logger()


class Node:
    def __init__(self, node_name: str, embedding: List[float] = []):
        self.name = node_name
        self.embeds = embedding
        self.created_at = int(time.time())

    def __repr__(self):
        return self.name


class Command:
    def __init__(self, name: str, description: str, endpoint: str, input_schema: str, output_schema: str):
        self.name = name
        self.description = description
        self.endpoint = endpoint
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.created_at = int(time.time())\


    def to_prompt(self) -> str:
        return f"""{self.name}: {self.description}
input must strictly be a json following this schema:
{self.input_schema}"""

    def __repr__(self):
        return f"{self.name}: {self.description}"

    def update_embeds(self, embeds: List[float]) -> None:
        self.embeds = embeds


class MemoryUnit:
    def __init__(self, role, content) -> None:
        assert role == "user" or role == "system" or role == "assistant"
        self.role = role
        self.content = content
        self.num_tokens = count_message_tokens({
            "role": role,
            "content": content
        })

    @classmethod
    def from_json(cls, data: Dict[str, str]):
        return cls(data["role"], data["content"])

    def to_json(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content
        }

    def __repr__(self) -> str:
        return f"<{self.role}>: {self.content}[{self.num_tokens} tokens]"

    def __str__(self) -> str:
        return f"<{self.role}>: {self.content}[{self.num_tokens} tokens]"


class Knowledge:

    def __init__(self, created_at: int, content: str, embeds: List = []):
        self.created_at = int(created_at)
        self.content = content
        self.embeds = embeds
        self.num_tokens = count_message_tokens({
            "content": content
        })

    @classmethod
    def from_json(cls, knowledge_json: Dict):
        embeds = []
        if "embeds" in knowledge_json:
            embeds = knowledge_json["embeds"]
        return cls(knowledge_json["created_at"], knowledge_json["content"], embeds)

    def to_json(self):
        return {
            "created_at": self.created_at,
            "content": self.content,
            "embeds": self.embeds
        }

    def _to_str(self, created_at: int, content: str) -> str:
        return f"knowledge created at {datetime.fromtimestamp(created_at)}. content: {content}"

    def __repr__(self):
        return f"knowledge({self.content}, created at {self.created_at})"

    def update_embeds(self, embeds: List[float]) -> None:
        self.embeds = embeds


class BaseMemory:
    MAX_NUM_TOKENS_CONTEXT = 1000
    MAX_NUM_TOKENS_MEMORY = 1000
    MAX_NUM_TOKENS_KNOWLEDGE = 200

    def __init__(self, main_agent_config: Dict[any, any], context: str, domain_knowledge: List[Knowledge], memory: List[MemoryUnit]):
        self._reflections = []
        self._commands = []
        self._domain_knowledge = []
        self._memory = []
        self._context = ""
        self.num_tokens_memory = 0
        self._memory = memory
        self.num_tokens_context = 0
        self.num_tokens_domain_knowledge = 0
        self._main_agent_config = main_agent_config
        # role: system. Serves as background information of the memory manager
        self.update_context(context)
        # role: system. When initializing, will be constructed from agentware.memory,
        # context, and domain knowledge storage
        self.update_knowledge(domain_knowledge)

    def update_context(self, context: str):
        self._context = context[:self.MAX_NUM_TOKENS_CONTEXT]
        self.num_tokens_context = count_message_tokens({
            "role": "system",
            "content": self._context
        })

    def get_main_agent_config(self) -> Dict[str, any]:
        return self._main_agent_config

    def get_num_tokens_memory(self):
        return self.num_tokens_memory

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

    def _compress_memory(self, reflect) -> List[str]:
        """ Compress memory when its too long
        """
        raise Exception("Not implemented")

    def prepare_run(self, prompt_prefix: str, prompt: str):
        self.add_memory({
            "role": "user",
            "content":  prompt
        })

    def add_memory(self, memory: Dict[str, str]):
        new_memory = MemoryUnit(memory["role"], memory["content"])
        self._memory.append(new_memory)
        self.num_tokens_memory += new_memory.num_tokens
        if self.num_tokens_memory > self.MAX_NUM_TOKENS_MEMORY:
            self._compress_memory(reflect=True)

    def update_knowledge(self, knowledges: List[Knowledge]):
        for k in knowledges:
            self.num_tokens_domain_knowledge += k.num_tokens
            if self.num_tokens_domain_knowledge > self.MAX_NUM_TOKENS_KNOWLEDGE:
                break

    def delete_memory(self, memory_index: int):
        if memory_index >= len(self._memory):
            logger.debug(
                f"Deleting index {memory_index} out of range of 0-{len(self._memory - 1)}")
        self.num_tokens_memory -= self._memory[memory_index].num_tokens
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
        prefix = f'\n************* Memory({self.num_tokens_memory + self.num_tokens_context + self.num_tokens_domain_knowledge} tokens) *************\n'
        suffix = f'\n************* End of Memory *************\n'
        context_str = f'\n<context> [{self.num_tokens_context} tokens]: {self._context}\n'
        knowledge_str = f"\n<knowledge>: [{self.num_tokens_domain_knowledge}]\n" + "\n----------------------\n".join(
            [k.__str__() for k in self.get_domain_knowledge()])
        memory_str = f"<memory> [{self.num_tokens_memory} tokens]:\n" + "\n----------------------\n".join(
            [m.__str__() for m in self.get_memory()])
        return prefix + context_str + knowledge_str + memory_str + suffix

    def __repr__(self):
        return self.__str__()


class BaseConnector:
    def __init__(self, config: Dict[str, str]):
        super().__init__()

    def get_token(self, api_key: str) -> str:
        raise Exception("Not iimplemented")

    def create_agent(self) -> int:
        raise Exception("Not iimplemented")

    def _get_command_hub_id(self) -> str:
        raise Exception("Not iimplemented")

    def _get_knowledge_base_id(self, agent_id: int):
        raise Exception("Not iimplemented")

    def list_agents(self):
        raise Exception("Not iimplemented")

    def get_longterm_memory(self, agent_id: int, page_number: int, page_size: int) -> List[Dict]:
        raise Exception("Not iimplemented")

    def update_longterm_memory(self, agent_id: int, memory_units: List[MemoryUnit]):
        raise Exception("Not iimplemented")

    def update_checkpoint(self, agent_id: int, agent_config: Dict[any, any], helper_agent_configs: Dict[str, Dict[any, any]], memory_units: List[MemoryUnit], knowledges: List[Knowledge], context: str):
        raise Exception("Not iimplemented")

    def get_checkpoint(self, agent_id: int) -> Tuple[Dict[any, any], Dict[str, Dict[any, any]], List[MemoryUnit], List[Knowledge], str]:
        raise Exception("Not iimplemented")

    def save_knowledge(self, agent_id: int, knowledges: List[Knowledge]):
        raise Exception("Not iimplemented")

    def search_commands(self, keyword: str, token_limit=100) -> List[Command]:
        raise Exception("Not iimplemented")

    def search_knowledge(self, agent_id: int, keyword: str, token_limit=100) -> List[Dict]:
        raise Exception("Not iimplemented")

    def get_recent_knowledge(self, agent_id: int, token_limit=100):
        raise Exception("Not implemented")

    def get_embeds(self, text: str):
        raise Exception("Not iimplemented")


class BaseAgent:
    """
        Base agent class with
        - config initialization
        - parse and retry
    """
    MODEL_NAME = "gpt-3.5-turbo"
    MAX_NUM_RETRIES = 3

    def __init__(self, cfg: Dict[str, any]):
        if not cfg:
            raise Exception("Invalid config")
        if (not cfg["name"]):
            raise Exception("config missing required entry {cfg}")
        self._config = cfg
        # output format
        if "output_format" in cfg:
            self.format_instruction = None
            if "instruction" in cfg["output_format"]:
                self.format_instruction = cfg["output_format"]["instruction"]
            self.output_schema = None
            if "output_schema" in cfg["output_format"]:
                self.output_schema = cfg["output_format"]["output_schema"]
            self.termination_observation = None
            if "termination_observation" in cfg["output_format"]:
                self.termination_observation = cfg["output_format"]["termination_observation"]
        if "prompt_prefix" in cfg:
            self.prompt_prefix = cfg["prompt_prefix"]

    def get_config(self) -> Dict[str, any]:
        return self._config

    def get_agent_graph_identifier(self):
        agent_id_str = ""
        if hasattr(self, "_agent_id") and self._agent_id:
            agent_id_str = self._agent_id
        return hash(f"{self._config['name']}:{self._config['conversation_setup']}:{agent_id_str}")

    def _messages_to_str(self, messages: List[Dict[str, str]]) -> str:
        message_prefix = "\n************* Conversation *************\n"
        message_suffix = "\n********* End of Conversation *********\n"
        message_str = "\n---------------------------------\n".join(
            [f"<{m['role']}>: {m['content']}" for m in messages])
        return message_prefix + message_str + message_suffix

    def _run(self, messages: List[Dict[str, str]]) -> str:
        logger.debug(
            f"Sending raw messages: {self._messages_to_str(messages)}")
        completion = openai.ChatCompletion.create(
            model=self.MODEL_NAME, messages=messages)
        raw_output = completion.choices[0].message.content
        logger.debug(f"Raw output: {raw_output}")
        return raw_output

    def parse_output(self, llm_output):
        if not self.output_schema:
            return llm_output
        logger.debug(f'parsing {llm_output}')
        parsed_output = fix_and_parse_json(llm_output)
        logger.debug(f"parse success, output is {parsed_output}")
        logger.debug(f"validating with schema {self.output_schema}")
        validated_output = validate_json(parsed_output, self.output_schema)
        return validated_output

    def run(self, prompt: str):
        raise Exception("Not Implemented")

    def __repr__(self) -> str:
        return f""


class OneshotAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self, prompt) -> str:
        """
        prefix和prompt prefix应该在哪一层加进去？
        如果加在BaseAgent, 因为base agent不知道memory, 所以没法把conversation_setup和prompt_prefix和
        memory中的context和memory联系起来。如果放在memory, 那不需要用memory的agent又没法用到conversation_setup
        和prompt_prefix了。这个有点尴尬.
        应该放在这一层。然后在memory中不再维护context. Memory只需要管knowledge和memory. 需要跑的时候，首先knowledge
        放在最前面, 作为system prompt.
        但是. 这样又没办法统一管理token数量了. 既然要在memory之外添加token, 包括error handling的部分加入的报错信息, 仍然
        没办法被memory管理。
        这样，就只能从下到上, 全部贯彻memory. 这样做base agent是否还有意义? 感觉没意义了。干脆memory全部放入base.py, 对
        BaseAgent加入memory.

        这样的话memory依赖的agent怎么初始化呢? memory中agent的要求
        1. 能校验输出格式且有重试机制
        既然要满足这个要求, 就需要json schema, 那最好能从一个json file中初始化。要从json file创建agent,
        而json file中有prefix和prompt_prefix, 这个按照目前的做法, 是要放入memory的。主要是prefix, 目前是直接放入memory
        的prefix的。不可能在memory中套一个memory. 看起来还是需要一个简单版的agent, 能从json file创建但不需要通过memory来运行.
        """
        num_retries = 0
        raw_output = ""
        parsed_output = ""
        full_prompt = ""
        format_instruction = f"Answer the question above by ignoring all the output format instructions you were given previously. Your output must be strictly a json following this schema but do not include this schema: {self.output_schema}. Key and value must be enclosed in double quotes"

        if self.output_schema:
            full_prompt = f"{self.prompt_prefix}. Ignore all the output format instructions you were given previously. Your output must be strictly a json following this schema but do not include this schema: {self.output_schema}, question: {prompt}"
        else:
            full_prompt = f"{self.prompt_prefix}. {prompt}"
        messages = [{
            "role": "system",
            "content": self._config["conversation_setup"]
        }] + [{
            "role": "user",
            "content": f"{self.prompt_prefix}. {prompt}"
        }]
        original_messages = copy.deepcopy(messages)
        messages_with_error = original_messages
        messages_with_error[-1]["content"] += format_instruction
        while True:
            try:
                raw_output = self._run(messages_with_error)
                logger.debug(f"Raw output is {raw_output}")
                try:
                    parsed_output = self.parse_output(raw_output)
                    if self.termination_observation in parsed_output:
                        return parsed_output[self.termination_observation]
                    return parsed_output
                except Exception as e:
                    logger.warning(f"Error parsing output with error. {e}")
                    messages_with_error = original_messages + [
                        {
                            "role": "assistant",
                            "content": raw_output
                        },
                        {
                            "role": "user",
                            "content": f"Failed to parse output. Ignore all the format instructions you were given previously. Your output must be a json that strictly follow the schema while not including it {self.output_schema}"
                        },
                    ]
            except Exception as e:
                logger.warning(f"Error running prompt with error {e}")
            if num_retries >= self.MAX_NUM_RETRIES:
                if num_retries >= self.MAX_NUM_RETRIES:
                    logger.debug("Max number of retries exceeded")
                break
            num_retries += 1
        return parsed_output


class BaseMilvusStore():
    DEFAULT_EMBED_DIM = 1536  # openai embedding

    def _create_collection(self, collection_name: str) -> Collection:
        raise Exception("Not Implemented")

    def __init__(self, cfg: dict[str, any]) -> None:
        if not cfg:
            raise BaseException("Invalid config")
        try:
            self.milvus_uri = cfg['uri']
            self.user = cfg['user']
            self.port = cfg['port']
            self.nprobe = int(cfg['nprobe'])
            self.nlist = int(cfg['nlist'])
        except ValueError as e:
            raise BaseException(f"Missing key in config caused error {e}")
        self.client = Milvus(self.milvus_uri, self.port)

        self.connect = connections.connect(
            alias="default",
            host=self.milvus_uri,
            port=self.port
        )
        logger.debug(f"Connected to vector store: {self.milvus_uri}")

    def check_and_maybe_create_collection(self, collection_name: str):
        if utility.has_collection(collection_name):
            return
        logger.debug(
            f"default collection {collection_name} not found. Creating one.")
        collection = self._create_collection(collection_name)
        logger.debug(
            f"Creating default collection: {collection_name}")

        collection.load()

    def _similarity_query(self, query_vector, collection_name, output_fields: List[str]):
        try:
            search_params = {"metric_type": "L2",
                             "params": {"nprobe": self.nprobe}}
            c = Collection(collection_name)
            results = c.search([query_vector],
                               anns_field="vector",
                               param=search_params,
                               round_decimal=-1,
                               output_fields=output_fields,
                               limit=999)
            return results
        except Exception as err:
            logger.debug("get err {}".format(err))
            return err


BASE_AGENT_CONFIGS_DIR = "src/base_agent_configs"


def create_base_agents() -> Dict[str, OneshotAgent]:
    facts_agent_config_path = f"{BASE_AGENT_CONFIGS_DIR}/facts.json"
    ref_q_agent_config_path = f"{BASE_AGENT_CONFIGS_DIR}/reflection_question.json"
    ref_agent_config_path = f"{BASE_AGENT_CONFIGS_DIR}/reflection.json"
    summarize_agent_config_path = f"{BASE_AGENT_CONFIGS_DIR}/summarize.json"
    tool_query_agent_config_path = f"{BASE_AGENT_CONFIGS_DIR}/tool_query.json"

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
