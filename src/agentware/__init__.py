import os
from dotenv import load_dotenv

load_dotenv()

endpoint = "http://localhost:8741"
openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("API_KEY")

HELPER_AGENT_CONFIGS_DIR_NAME = "base_agent_configs"
