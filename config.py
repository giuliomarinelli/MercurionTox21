import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values
from pydantic import ValidationError
from schemas.schemas import Configuration


class ConfigurationError(ValueError):
    """Invalid startup settings; never include supplied values in diagnostics."""


ENVIRONMENTS = ("development", "test", "staging", "production")


def get_config(
    environment: Mapping[str, str] | None = None, *, root: Path | None = None
) -> Configuration:
    process = dict(os.environ if environment is None else environment)
    py_env = process.get("PY_ENV", "development")
    if py_env not in ENVIRONMENTS:
        raise ConfigurationError("PY_ENV must be development, test, staging or production")

    # Local dotenv files are convenience input only; never read them in deployed modes.
    values: dict[str, str | None] = {}
    if py_env in ("development", "test"):
        values.update(dotenv_values((root or Path.cwd()) / f".env.{py_env}", interpolate=False))
    values.update(process)
    raw = {
        "py_env": py_env,
        "nats_url": values.get("NATS_URL"),
        "version": values.get("VERSION"),
        "model_package_dir": values.get("MODEL_PACKAGE_DIR"),
        "jwt_public_key_path": values.get("JWT_PUBLIC_KEY_PATH")
        or ("public.pem" if py_env == "production" else f"public.{py_env}.pem"),
    }
    try:
        return Configuration(**raw)
    except ValidationError as error:
        fields = sorted({str(item["loc"][0]) for item in error.errors(include_input=False)})
        raise ConfigurationError("Invalid configuration fields: " + ", ".join(fields)) from None


def nats_subject(base: str, environment: str) -> str:
    if environment not in ENVIRONMENTS:
        raise ConfigurationError("Invalid NATS environment")
    return base if environment == "production" else f"{environment}.{base}"


if __name__ == "__main__":
    print(f"Configuration valid for {get_config().py_env}")
