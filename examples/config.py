"""Application configuration and environment setup."""

import os
import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """Application configuration."""

    google_api_key: str
    langsmith_tracing: bool
    langsmith_endpoint: str
    langsmith_api_key: Optional[str]
    langsmith_project: str
    log_level: int = logging.INFO


def load_config() -> Config:
    """Load configuration from environment variables.

    Returns:
        Config: Application configuration.

    Raises:
        ValueError: If required environment variables are missing.
    """
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    if langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", "true")

    return Config(
        google_api_key=google_api_key,
        langsmith_tracing=langsmith_tracing,
        langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "react-agent-compensation-demo"),
        log_level=logging.INFO,
    )


def configure_logging(config: Config) -> logging.Logger:
    """Configure application logging.

    Suppresses verbose library logs while keeping application logs visible.

    Args:
        config: Application configuration.

    Returns:
        Logger: Configured application logger.
    """
    # Suppress verbose library logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langchain_core").setLevel(logging.WARNING)
    logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)
    logging.getLogger("langgraph").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)

    # Configure root logger with minimal format
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Application logger
    logger = logging.getLogger("compensation_demo")
    logger.setLevel(config.log_level)

    # Compensation middleware logger for observability
    comp_logger = logging.getLogger("react_agent_compensation")
    comp_logger.setLevel(logging.INFO)

    return logger
