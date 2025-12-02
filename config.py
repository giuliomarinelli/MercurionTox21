import os
from dotenv import load_dotenv

def env_file_for(py_env: str | None) -> str:
    py_env = (py_env or "development").lower()
    return {
        "development": ".env.development",
        "test": ".env.test",
        "staging": ".env.staging",
        "production": ".env",
        "prod": ".env",
    }.get(py_env, ".env.development")

PY_ENV = os.getenv("PY_ENV", "development")
ENV_FILE = env_file_for(PY_ENV)

load_dotenv(ENV_FILE, override=False)  

DB_URL = os.getenv("DB_URL")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
