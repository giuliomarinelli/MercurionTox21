import os
from dotenv import load_dotenv
from pydantic import ValidationError
from schemas.schemas import Configuration
import sys

def get_config() -> Configuration:
    
    def env_file_for(py_env: str | None) -> str:
        py_env = (py_env or "development").lower()
        return {
            "development": ".env.development",
            "test": ".env.test",
            "staging": ".env.staging",
            "production": ".env",            
        }.get(py_env, ".env.development")


    PY_ENV = os.getenv("PY_ENV", "development")
    ENV_FILE = env_file_for(PY_ENV)
    load_dotenv(ENV_FILE, override=False)  
    raw = {
        "nats_url": os.getenv("NATS_URL"),
        "version": os.getenv("VERSION"),
        "py_env": PY_ENV
    }
    try:
        return Configuration(**raw)
    except (ValidationError, Exception) as e:        
        print(f"FATAL: invalid configuration\n{e}", file=sys.stderr)
        sys.exit(1)
        
