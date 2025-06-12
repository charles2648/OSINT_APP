# Contains logic to select and instantiate LLMs dynamically.

import os
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

AVAILABLE_MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "gpt-4o-mini": {
        "provider": "OpenAI", "model_name": "gpt-4o-mini",
        "api_key_env_var": "OPENAI_API_KEY", "strengths": "Fast, highly capable, cost-effective.",
    },
    "gpt-4-turbo": {
        "provider": "OpenAI", "model_name": "gpt-4-turbo",
        "api_key_env_var": "OPENAI_API_KEY", "strengths": "Top-tier reasoning for complex tasks.",
    },
    "claude-3-haiku-20240307": {
        "provider": "Anthropic", "model_name": "claude-3-haiku-20240307",
        "api_key_env_var": "ANTHROPIC_API_KEY", "strengths": "Extremely fast, good for summarization.",
    },
    "openrouter/deepseek/deepseek-chat": {
        "provider": "OpenRouter", "model_name": "deepseek/deepseek-chat",
        "api_key_env_var": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
        "strengths": "Access to various models via a single API key.",
    },
}

def get_llm_instance(model_id: str, temperature: float = 0.7) -> BaseChatModel:
    config = AVAILABLE_MODELS_CONFIG.get(model_id)
    if not config:
        raise ValueError(f"Model ID '{model_id}' is not configured or supported.")

    provider = config["provider"]
    model_name = config["model_name"]
    api_key_env_var = config.get("api_key_env_var")
    
    if not api_key_env_var:
        raise ValueError(f"API key environment variable not defined for model {model_id}")

    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(f"API key '{api_key_env_var}' not found in environment variables.")

    if provider == "OpenAI":
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
    elif provider == "Anthropic":
        return ChatAnthropic(model=model_name, temperature=temperature, api_key=api_key)
    elif provider == "OpenRouter":
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=config.get("base_url"),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")