import redis


class DbClient:
    def __init__(self):
        self.redis_host = "127.0.0.1"
        self.redis_port = 6379
        self.redis_password = None

        self.client = redis.Redis(
            host=self.redis_host, port=self.redis_port, password=self.redis_password)
