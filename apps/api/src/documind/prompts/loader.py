from functools import lru_cache
from pathlib import Path


PROMPTS_DIR = Path(__file__).parent


@lru_cache
def load_prompt(name: str, version: str = "v1") -> str:
    """Load a named prompt revision so prompt changes are explicit and reviewable."""
    path = PROMPTS_DIR / version / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown prompt revision: {version}/{name}")
    return path.read_text(encoding="utf-8").strip()
