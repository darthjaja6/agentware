import time
import json
import requests
import os
import openai

from typing import Dict, List, Tuple
from agentware.base import BaseConnector, Knowledge, Command, MemoryUnit
from agentware.agent_logger import Logger

logger = Logger()


class Connector(BaseConnector):
    """

    modes
    - Use an agent, only have access to newly created
    - Fork an agent that allows access, have access to prefix, prompt prefix and knowledge
    -
    """

    def __init__(self, config: Dict[str, str]):
        assert config
        super().__init__(config)
        self._agent_id = None
        self._config = config
        self.endpoint = config["endpoint"]
        self._token = ""
        self.api_key = config["api_key"]
        self.get_token()

    def get_token(self) -> str:
        # URL to send the request to
        url = os.path.join(self.endpoint, "get_token", self.api_key)
        # Send GET request
        response = requests.get(url)
        # Check the response status code
        if response.status_code == 200:

            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            self._token = data["token"]
            return self._token
        else:
            # Request failed
            logger.debug(
                f'Request failed with status code: {response.status_code}')
            return None

    def create_agent(self) -> int:
        # URL to send the request to
        url = os.path.join(self.endpoint, "create_agent")
        headers = {
            'Authorization': f'Bearer {self._token}'
        }
        # Send GET request
        response = requests.put(url, headers=headers)
        # Check the response status code
        if response.status_code == 200:

            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            return data["agent_id"]
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

    def list_agents(self):
        # URL to send the request to
        url = os.path.join(self.endpoint, "list_agents")
        headers = {
            'Authorization': f'Bearer {self._token}'
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
        url = os.path.join(self.endpoint, "get_longterm_memory", str(agent_id))
        headers = {
            'Authorization': f'Bearer {self._token}'
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
            self.endpoint, "update_longterm_memory", str(agent_id))
        headers = {
            'Authorization': f'Bearer {self._token}'
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

    def update_checkpoint(self, agent_id: int, agent_config: Dict[any, any], helper_agent_configs: Dict[str, Dict[any, any]], memory_units: List[MemoryUnit], knowledges: List[Knowledge], context: str):
        if agent_id is None:
            agent_id = -1
        url = os.path.join(
            self.endpoint, "update_checkpoint", str(agent_id))
        headers = {
            'Authorization': f'Bearer {self._token}'
        }
        ckpt_data = json.dumps({
            "agent_config": agent_config,
            "helper_agent_configs": helper_agent_configs,
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

    def get_checkpoint(self, agent_id: int) -> Tuple[Dict[any, any], Dict[str, Dict[any, any]], List[MemoryUnit], List[Knowledge], str]:
        # URL to send the request to
        url = os.path.join(self.endpoint, "get_checkpoint", str(agent_id))
        # Send GET request
        headers = {
            'Authorization': f'Bearer {self._token}'
        }
        response = requests.get(url, headers=headers)
        # Check the response status code
        if response.status_code == 200:
            logger.debug(f'Request to {url} was successful')
            data = json.loads(response.text)
            logger.debug(f'agent data is {data}')
            main_agent_config = data["main_agent_config"]
            helper_agent_configs = data["helper_agent_configs"]
            memory_units = [MemoryUnit.from_json(
                m) for m in data["memory_units"]]
            knowledges = [Knowledge.from_json(k) for k in data["knowledges"]]
            context = data["context"]
            return main_agent_config, helper_agent_configs, memory_units, knowledges, context
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
        url = os.path.join(self.endpoint, "save_knowledge",
                           knowledge_base_identifier)
        # Send GET request
        headers = {
            'Authorization': f'Bearer {self._token}'
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
        url = os.path.join(self.endpoint, "search_commands",
                           self._get_command_hub_id())
        # Send GET request
        headers = {
            'Authorization': f'Bearer {self._token}'
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
        url = os.path.join(self.endpoint, "search_knowledge",
                           self._get_knowledge_base_id(agent_id))
        # Send GET request
        headers = {
            'Authorization': f'Bearer {self._token}'
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
        url = os.path.join(self.endpoint, "get_recent_knowledge",
                           self._get_knowledge_base_id(agent_id))
        # Send GET request
        headers = {
            'Authorization': f'Bearer {self._token}'
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
        url = os.path.join(self.endpoint, "search_kg",
                           self._get_knowledge_graph_label(agent_id))
        # Send GET request
        headers = {
            'Authorization': f'Bearer {self._token}'
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
