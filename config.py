import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.getenv(name, default)

    if value is None:
        if required:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default

    value = str(value).strip()
    if value == "" and required:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value
