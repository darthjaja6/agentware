import redis
from agentware.core_engines import CoreEngineBase


class EchoCoreEngine(CoreEngineBase):
    def run(self, prompt):
        return f"An echo of {prompt}"


class DbClient:
    def __init__(self):
        self.redis_host = "localhost"
        self.redis_port = 6379

        self.client = redis.Redis(
            host=self.redis_host, port=self.redis_port)
