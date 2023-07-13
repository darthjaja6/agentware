import os
import time
import json
import requests
import openai
import agentware
import copy

from typing import List, Dict, Tuple
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from pymilvus import Milvus, connections, utility
from agentware.agent_logger import Logger
from datetime import datetime
from agentware.utils.num_token_utils import count_message_tokens
from typing import List, Dict
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


class BaseAgent:
    """
        Base agent class with
        - config initialization
        - parse and retry
    """
    MODEL_NAME = "gpt-3.5-turbo"
    MAX_NUM_RETRIES = 3

    def __init__(self):
        self._config = None

    @classmethod
    def init(cls, cfg: Dict[str, any]):
        agent = cls()
        if not cfg:
            raise Exception("Invalid config")
        if (not cfg["name"]):
            raise Exception("config missing required entry {cfg}")
        agent._config = cfg
        # output format
        if "output_format" in cfg:
            agent.format_instruction = None
            if "instruction" in cfg["output_format"]:
                agent.format_instruction = cfg["output_format"]["instruction"]
            agent.output_schema = None
            if "output_schema" in cfg["output_format"]:
                agent.output_schema = cfg["output_format"]["output_schema"]
            agent.termination_observation = None
            if "termination_observation" in cfg["output_format"]:
                agent.termination_observation = cfg["output_format"]["termination_observation"]
        if "prompt_prefix" in cfg:
            agent.prompt_prefix = cfg["prompt_prefix"]
        return agent

    def get_config(self) -> Dict[str, any]:
        return self._config

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
        super().__init__()
        self._config = cfg

    def run(self, prompt) -> str:
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


class Connector():
    """

    modes
    - Use an agent, only have access to newly created
    - Fork an agent that allows access, have access to prefix, prompt prefix and knowledge
    -
    """

    def __init__(self):
        self._agent_id = None

    def verify_endpoint(self):
        if not agentware.endpoint:
            raise ValueError(
                f"Invalid agentware endpoint {agentware.endpoint}")

    def register_agent(self, agent_id: str) -> int:
        # URL to send the request tor
        url = os.path.join(agentware.endpoint, "register_agent")
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        data = json.dumps({
            "agent_id": agent_id
        })
        # Send GET request
        response = requests.put(url, headers=headers, data=data)
        # Check the response status code
        if response.status_code == 200:
            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            return data["exists"]
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            return None

    def _get_command_hub_id(self) -> str:
        return "command_collection"

    def _get_knowledge_base_id(self, agent_id: int):
        return f"knowledge_{agent_id}"

    def _get_knowledge_graph_label(self, agent_id: int):
        return f"knowledge_graph_{agent_id}"

    def all_agents(self) -> List[str]:
        url = os.path.join(
            agentware.endpoint, "all_agents")
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        # Send GET request
        response = requests.get(url, headers=headers)
        # Check the response status code
        if response.status_code == 200:
            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            return data
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            return None

    def get_longterm_memory(self, agent_id: int, page_number: int, page_size: int) -> List[Dict]:
        url = os.path.join(
            agentware.endpoint, "get_longterm_memory", str(agent_id))
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        params = {
            'page_number': page_number,
            'page_size': page_size
        }
        # Send GET request
        response = requests.get(url, headers=headers, params=params)
        # Check the response status code
        if response.status_code == 200:

            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            memory_units = [MemoryUnit.from_json(d) for d in data]
            return memory_units
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            return None

    def update_longterm_memory(self, agent_id: int, memory_units: List[MemoryUnit]):
        url = os.path.join(
            agentware.endpoint, "update_longterm_memory", str(agent_id))
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        memory_data = json.dumps({
            "memory_data": [m.to_json() for m in memory_units]
        })
        # Send GET request
        response = requests.put(url, headers=headers, data=memory_data)
        # Check the response status code
        if response.status_code == 200:

            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            return data
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            return None

    def update_checkpoint(self, agent_id: int, agent_config: Dict[any, any], memory_units: List[MemoryUnit], knowledges: List[Knowledge], context: str):
        if agent_id is None:
            agent_id = -1
        url = os.path.join(
            agentware.endpoint, "update_checkpoint", str(agent_id))
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        ckpt_data = json.dumps({
            "agent_config": agent_config,
            "memory_data":  [m.to_json() for m in memory_units],
            "knowledge": [k.to_json() for k in knowledges],
            "context": context
        })
        logger.info(
            f"Saving checkpoint {ckpt_data}")
        # Send GET request
        response = requests.put(url, headers=headers, data=ckpt_data)
        # Check the response status code
        if response.status_code == 200:

            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            return data
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            return None

    def get_checkpoint(self, agent_id: str) -> Tuple[Dict[any, any], Dict[str, Dict[any, any]], List[MemoryUnit], List[Knowledge], str]:
        # URL to send the request to
        url = os.path.join(agentware.endpoint, "get_checkpoint")
        # Send GET request
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        params = {
            "agent_id": agent_id
        }
        response = requests.get(url, headers=headers, params=params)
        # Check the response status code
        if response.status_code == 200:
            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            print("data is", data)
            if not data["success"]:
                raise ValueError(data["error_code"])
            main_agent_config = data["main_agent_config"]
            memory_units = [MemoryUnit.from_json(
                m) for m in data["memory_units"]]
            knowledges = [Knowledge.from_json(
                k) for k in data["knowledges"]]
            context = data["context"]
            return main_agent_config, memory_units, knowledges, context
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            return None

    def save_knowledge(self, agent_id: int, knowledges: List[Knowledge]):
        knowledge_base_identifier = self._get_knowledge_base_id(agent_id)
        logger.info(
            f"Saving knowledge: {knowledges} to knowledge base {knowledge_base_identifier}")
        for i, knowledge in enumerate(knowledges):
            if knowledge.embeds:
                continue
            embeds = self.get_embeds(knowledge.content)
            knowledges[i].update_embeds(embeds)
        # URL to send the request to
        url = os.path.join(agentware.endpoint, "save_knowledge",
                           knowledge_base_identifier)
        # Send GET request
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        data = json.dumps({
            "knowledges": [k.to_json() for k in knowledges]
        })
        response = requests.put(url, headers=headers, data=data)
        # Check the response status code
        if response.status_code == 200:

            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            return data
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            return None

    def search_commands(self, keyword: str, token_limit=100) -> List[Command]:
        query_embeds = self.get_embeds(keyword)
        # URL to send the request to
        url = os.path.join(agentware.endpoint, "search_commands",
                           self._get_command_hub_id())
        # Send GET request
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        data = json.dumps({
            "query_embeds": query_embeds,
            "token_limit": token_limit
        })
        response = requests.get(url, headers=headers, data=data)
        # Check the response status code
        if response.status_code == 200:

            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            return data
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            logger.debug(response.text)
            return None

    def search_knowledge(self, agent_id: int, keyword: str, token_limit=100) -> List[Knowledge]:
        query_embeds = self.get_embeds(keyword)
        # URL to send the request to
        url = os.path.join(agentware.endpoint, "search_knowledge",
                           self._get_knowledge_base_id(agent_id))
        # Send GET request
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        data = json.dumps({
            "query_embeds": query_embeds,
            "token_limit": token_limit
        })
        response = requests.get(url, headers=headers, data=data)
        # Check the response status code
        if response.status_code == 200:
            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            logger.debug(f"knowledge data is {data}")
            return [Knowledge.from_json(knowledge_json) for knowledge_json in data]
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            logger.debug(response.text)
            return None

    def get_recent_knowledge(self, agent_id: int, token_limit=100) -> List[Knowledge]:
        url = os.path.join(agentware.endpoint, "get_recent_knowledge",
                           self._get_knowledge_base_id(agent_id))
        # Send GET request
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        data = json.dumps({
            "token_limit": token_limit
        })
        response = requests.get(url, headers=headers, data=data)
        # Check the response status code
        if response.status_code == 200:
            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            logger.debug(f"knowledge data is", data)
            return data
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            logger.debug(response.text)
            return None

    def get_embeds(self, text: str):
        model = "text-embedding-ada-002"
        text = text.replace("\n", " ")
        logger.debug(f"Getting embedding of: {text}")
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

    def search_kg(self, agent_id: int, keyword: str, token_limit=100) -> List[Knowledge]:
        query_embeds = self.get_embeds(keyword)
        # URL to send the request to
        url = os.path.join(agentware.endpoint, "search_kg",
                           self._get_knowledge_graph_label(agent_id))
        # Send GET request
        headers = {
            'Authorization': f'Bearer {agentware.api_key}'
        }
        data = json.dumps({
            "query_embeds": query_embeds,
            "token_limit": token_limit
        })
        response = requests.get(url, headers=headers, data=data)
        # Check the response status code
        if response.status_code == 200:
            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            logger.debug(f"knowledge data is {data}")
            return [Knowledge.from_json(knowledge_json) for knowledge_json in data]
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            logger.debug(response.text)
            return None
