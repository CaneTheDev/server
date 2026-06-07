import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve paths relative to this file to correctly find the .env in backend/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ALLOWED_ORIGINS_RAW: str = os.getenv("ALLOWED_ORIGINS", "")
    
    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        if not self.ALLOWED_ORIGINS_RAW:
            return []
        if self.ALLOWED_ORIGINS_RAW == "*":
            return ["*"]
        return [
            origin.strip().strip("`").strip("'").strip('"') 
            for origin in self.ALLOWED_ORIGINS_RAW.split(",") 
            if origin.strip()
        ]

    OPENROUTER_API_KEY_ELIGIBILITY: str = os.getenv("OPENROUTER_API_KEY_ELIGIBILITY", "").strip()
    OPENROUTER_API_KEY_NETWORKING: str = os.getenv("OPENROUTER_API_KEY_NETWORKING", "").strip()
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "").strip()
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "").strip()

    OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    GITHUB_SEARCH_URL: str = "https://api.github.com/search/users"

    # Target Models
    MODEL_ELIGIBILITY: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    MODEL_NETWORKING: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    MODEL_PRIMARY: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    MODEL_ULTRA: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

    # Fallbacks in case primary free models fail
    MODEL_ELIGIBILITY_FALLBACK: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    MODEL_NETWORKING_FALLBACK: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

settings = Settings()
