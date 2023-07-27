import redis
import random
import os
import json

from agentware.core_engines import CoreEngineBase
from agentware import EMBEDDING_DIM
from typing import List

EMBEDS_FNAME = "data/embeds.json"


class FakeCoreEngine(CoreEngineBase):
    def __init__(self) -> None:
        super().__init__()
        embeds_fpath = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), EMBEDS_FNAME)
        with open(embeds_fpath, "r") as f:
            self._embeddings = json.loads(f.read())

    def get_embeds(self, text: str):
        if not text in self._embeddings:
            raise KeyError(f"Failed to get embedding for {text}")
        return self._embeddings[text]

    def get_sentences(self):
        return self._embeddings.keys()

    def run(self, prompt):
        return f"An echo of {prompt}"


class DbClient:
    def __init__(self):
        self.redis_host = "localhost"
        self.redis_port = 6379

        self.client = redis.Redis(
            host=self.redis_host, port=self.redis_port)
