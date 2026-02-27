from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # xAI (shared key for chat LLM + image generation)
    XAI_API_KEY: str = ""
    XAI_BASE_URL: str = "https://api.x.ai/v1"
    XAI_MODEL: str = "grok-4-fast-reasoning"
    XAI_IMAGE_MODEL: str = "grok-imagine-image"

    # ComfyUI (direct connection)
    COMFYUI_SERVER_ADDRESS: str = "127.0.0.1:8188"

    # Testing mode: use AIO.json workflow for all tasks (set to true to enable)
    USE_AIO_MODE: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
