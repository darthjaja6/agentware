import redis
import json

from typing import List, Dict, Tuple

LONG_MEMORY_KEY = "memory"
AGENT_KEY = "agent"
USER_KEY = "user"
API_INDEX_KEY = "api_key_index"  # api_key -> user_id
USER_AGENT_INDEX_KEY = "user_agent_index"  # user_id -> agent_id

AGENT_ID_COUNTER_KEY = "agent_id_counter"
USER_ID_COUNTER_KEY = "user_id_counter"


class DbClient:
    def __init__(self, config: Dict[str, str]):
        if not config:
            raise BaseException("Invalid db client config")
        self.AGENT_DB_ID = 0
        self.MEMORY_DB_ID = 1
        self.USER_DB_ID = 3

        # Establish a connection to the Redis server
        self.redis_host = config['ip']
        self.redis_port = int(config['port'])
        self.redis_password = None

        self.agent_client = redis.Redis(
            host=self.redis_host, port=self.redis_port, db=self.AGENT_DB_ID, password=self.redis_password)

        self.memory_client = redis.Redis(
            host=self.redis_host, port=self.redis_port, db=self.MEMORY_DB_ID, password=self.redis_password)

        self.user_client = redis.Redis(
            host=self.redis_host, port=self.redis_port, db=self.USER_DB_ID, password=self.redis_password)

    def get_agents_of_user(self, user_id: int):
        agent_ids_json = self.user_client.hget(
            f"{USER_AGENT_INDEX_KEY}", user_id)
        if not agent_ids_json:
            return []
        agent_ids = json.loads(agent_ids_json)
        if not agent_ids:
            return []
        agent_datas = []
        # Retrieve item names using the retrieved item IDs
        for agent_id in agent_ids:
            agent_data_json = self.agent_client.hgetall(
                f"{AGENT_KEY}:{agent_id}")
            if agent_data_json is not None:
                agent_data = {k.decode("utf8"): v.decode(
                    "utf8") for k, v in agent_data_json.items()}
                agent_data["id"] = agent_id
                agent_datas.append(agent_data)
        return agent_datas

    def _create_auto_incr_entry(self, client: redis.Redis, key: str, counter_key: str, value: Dict[any, any]):
        current_entity_id = None
        # Start a Redis transaction
        with client.pipeline() as pipe:
            while True:
                try:
                    # Watch the counter key to detect changes
                    pipe.watch(counter_key)
                    # Get the current value of the counter
                    current_entity_id = pipe.get(counter_key)
                    if current_entity_id is None:
                        # If the counter doesn't exist, initialize it to 1
                        current_entity_id = 1
                    else:
                        # Increment the counter by 1
                        current_entity_id = int(current_entity_id) + 1
                    # Update the counter key with the new value
                    pipe.multi()
                    pipe.set(counter_key, current_entity_id)
                    # Insert user information with the generated ID
                    pipe.hmset(f'{key}:{current_entity_id}', value)
                    # Execute the transaction
                    pipe.execute()
                    # Break out of the loop since the transaction was successful
                    break
                except redis.WatchError:
                    # Another process modified the  key during the transaction
                    continue
        return current_entity_id

    def create_agent(self, user_id: int) -> int:
        agent_id = self._create_auto_incr_entry(
            self.agent_client, AGENT_KEY, AGENT_ID_COUNTER_KEY, {"user_id": user_id})
        # Pushing agent id into user index
        agent_ids_json = self.user_client.hget(
            f"{USER_AGENT_INDEX_KEY}", user_id)
        if not agent_ids_json:
            agent_ids = [agent_id]
        else:
            agent_ids = json.loads(agent_ids_json)
            print(agent_ids)
            if agent_ids:
                agent_ids.extend([agent_id])
                agent_ids = list(set(agent_ids))
            else:
                agent_ids = [agent_id]
        self.user_client.hset(
            f"{USER_AGENT_INDEX_KEY}", user_id, json.dumps(agent_ids))
        return agent_id

    def create_user(self, config: Dict[str, str]) -> int:
        user_id = self._create_auto_incr_entry(
            self.user_client, USER_KEY, USER_ID_COUNTER_KEY, config)
        assert "api_key" in config
        api_key = config["api_key"]
        self.user_client.hset(API_INDEX_KEY, api_key, user_id)
        return user_id

    def get_checkpoint(self, agent_id: int) -> Tuple[Dict[any, any], Dict[str, any], List[Dict], List[Dict], str]:
        memory_checkpoint_json = self.memory_client.hget(
            AGENT_KEY, agent_id)
        if not memory_checkpoint_json:
            return dict(), dict(), [], [], ""
        memory_checkpoint = json.loads(memory_checkpoint_json)
        return memory_checkpoint["agent_config"], memory_checkpoint["helper_agent_configs"], memory_checkpoint["memory"], memory_checkpoint["knowledge"], memory_checkpoint["context"]

    def update_checkpoint(self, agent_config: Dict[any, any], helper_agents_configs: Dict[str, Dict[any, any]], memory_units: List[Dict], knowledges: List[Dict], context: str, user_id: int, agent_id: int = -1):
        if agent_id < 0:
            agent_id = self.create_agent(user_id)
        # Working memory is completely replaced
        working_memory = {
            "agent_config": agent_config,
            "helper_agent_configs": helper_agents_configs,
            "context": context,
            "knowledge": knowledges,
            "memory": memory_units
        }
        serialized_working_memory = json.dumps(working_memory)
        # Push each serialized dog to the list
        self.memory_client.hset(
            AGENT_KEY, agent_id, serialized_working_memory)

    def save_longterm_memory(self, agent_id: int, memory_units: List[Dict]):
        transaction = self.memory_client.pipeline()
        for m in memory_units:
            serialized_memory = json.dumps(m)
            # Push each serialized dog to the list
            transaction.rpush(
                f"{LONG_MEMORY_KEY}:{agent_id}", serialized_memory)
        # Execute the Redis transaction
        transaction.execute()

    def get_longterm_memory(self, agent_id: int, page_number, page_size) -> List[Dict]:
        start_index = page_number * page_size
        end_index = start_index + page_size - 1
        print(f"{start_index} -> {end_index}")
        memory_values = self.memory_client.lrange(
            f"{LONG_MEMORY_KEY}:{agent_id}", start_index, end_index)
        print("memory values is", memory_values)
        return [json.loads(m) for m in memory_values]
