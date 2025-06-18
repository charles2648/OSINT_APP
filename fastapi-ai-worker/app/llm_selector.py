# Contains logic to select and instantiate LLMs dynamically.

import os
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

AVAILABLE_MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    # OpenAI Models - Top Performers for Research & Reasoning
    "gpt-4o": {
        "provider": "OpenAI", "model_name": "gpt-4o",
        "api_key_env_var": "OPENAI_API_KEY", 
        "strengths": "Most capable model, exceptional at complex reasoning, analysis, and multi-step research tasks.",
        "reasoning_score": 9.5,
        "research_score": 9.4,
        "category": "Research Premium"
    },
    "gpt-4-turbo": {
        "provider": "OpenAI", "model_name": "gpt-4-turbo",
        "api_key_env_var": "OPENAI_API_KEY", 
        "strengths": "Top-tier reasoning for complex tasks, 128k context window ideal for deep research.",
        "reasoning_score": 9.3,
        "research_score": 9.2,
        "category": "Research Premium"
    },
    "gpt-4o-mini": {
        "provider": "OpenAI", "model_name": "gpt-4o-mini",
        "api_key_env_var": "OPENAI_API_KEY", 
        "strengths": "Strong reasoning capabilities with faster processing, excellent for iterative research.",
        "reasoning_score": 8.7,
        "research_score": 8.5,
        "category": "Research Balanced"
    },
    
    # Anthropic Models - Excellence in Analysis & Research
    "claude-3-5-sonnet-20241022": {
        "provider": "Anthropic", "model_name": "claude-3-5-sonnet-20241022",
        "api_key_env_var": "ANTHROPIC_API_KEY", 
        "strengths": "Outstanding analytical reasoning, excels at connecting disparate information in OSINT research.",
        "reasoning_score": 9.4,
        "research_score": 9.5,
        "category": "Research Premium"
    },
    "claude-3-opus-20240229": {
        "provider": "Anthropic", "model_name": "claude-3-opus-20240229",
        "api_key_env_var": "ANTHROPIC_API_KEY", 
        "strengths": "Most capable Claude model, superior at complex multi-step reasoning and deep analysis.",
        "reasoning_score": 9.3,
        "research_score": 9.1,
        "category": "Research Premium"
    },
    "claude-3-sonnet-20240229": {
        "provider": "Anthropic", "model_name": "claude-3-sonnet-20240229",
        "api_key_env_var": "ANTHROPIC_API_KEY", 
        "strengths": "Strong reasoning with excellent balance of speed and analytical depth.",
        "reasoning_score": 8.8,
        "research_score": 8.7,
        "category": "Research Balanced"
    },
    
    # Google Models - Advanced Reasoning Capabilities
    "google/gemini-pro-1.5": {
        "provider": "OpenRouter", "model_name": "google/gemini-pro-1.5",
        "api_key_env_var": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
        "strengths": "Google's most advanced model, exceptional at complex reasoning and large context understanding.",
        "reasoning_score": 9.1,
        "research_score": 9.0,
        "category": "Research Premium"
    },
    "google/gemini-flash-1.5": {
        "provider": "OpenRouter", "model_name": "google/gemini-flash-1.5",
        "api_key_env_var": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
        "strengths": "Fast processing with strong reasoning, good for rapid iterative research cycles.",
        "reasoning_score": 8.5,
        "research_score": 8.4,
        "category": "Research Balanced"
    },
    
    # Elite Open Source Models - Research & Reasoning Champions
    "meta-llama/llama-3.1-405b-instruct": {
        "provider": "OpenRouter", "model_name": "meta-llama/llama-3.1-405b-instruct",
        "api_key_env_var": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
        "strengths": "Meta's largest model, rivals GPT-4 in complex reasoning and research capabilities.",
        "reasoning_score": 9.0,
        "research_score": 8.8,
        "category": "Research Premium Open Source"
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "provider": "OpenRouter", "model_name": "meta-llama/llama-3.1-70b-instruct",
        "api_key_env_var": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
        "strengths": "Strong reasoning and analytical capabilities, excellent for detailed research tasks.",
        "reasoning_score": 8.6,
        "research_score": 8.5,
        "category": "Research Balanced Open Source"
    },
    "deepseek/deepseek-v2.5": {
        "provider": "OpenRouter", "model_name": "deepseek/deepseek-v2.5",
        "api_key_env_var": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
        "strengths": "Exceptional at complex reasoning and mathematical thinking, strong research capabilities.",
        "reasoning_score": 8.8,
        "research_score": 8.6,
        "category": "Research Premium Open Source"
    },
    "qwen/qwen-2.5-72b-instruct": {
        "provider": "OpenRouter", "model_name": "qwen/qwen-2.5-72b-instruct",
        "api_key_env_var": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
        "strengths": "Alibaba's top model, excellent multilingual reasoning and cross-cultural research capabilities.",
        "reasoning_score": 8.7,
        "research_score": 8.8,
        "category": "Research Balanced Open Source"
    },
    "mistralai/mixtral-8x22b-instruct": {
        "provider": "OpenRouter", "model_name": "mistralai/mixtral-8x22b-instruct",
        "api_key_env_var": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
        "strengths": "Advanced mixture of experts architecture, strong at complex analytical reasoning.",
        "reasoning_score": 8.5,
        "research_score": 8.4,
        "category": "Research Balanced Open Source"
    }
}

def get_llm_instance(model_id: str, temperature: float = 0.7) -> BaseChatModel:
    """
    Get an LLM instance for the specified model ID
    
    Args:
        model_id: The model identifier from AVAILABLE_MODELS_CONFIG
        temperature: The temperature setting for the model (0.0-1.0)
    
    Returns:
        BaseChatModel: The instantiated language model
    
    Raises:
        ValueError: If model_id is not supported or API key is missing
    """
    config = AVAILABLE_MODELS_CONFIG.get(model_id)
    if not config:
        available_models = list(AVAILABLE_MODELS_CONFIG.keys())
        raise ValueError(
            f"Model ID '{model_id}' is not configured or supported. "
            f"Available models: {available_models}"
        )

    provider = config["provider"]
    model_name = config["model_name"]
    api_key_env_var = config.get("api_key_env_var")
    
    if not api_key_env_var:
        raise ValueError(f"API key environment variable not defined for model {model_id}")

    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(
            f"API key '{api_key_env_var}' not found in environment variables. "
            f"Please set the environment variable for {provider}."
        )
    
    # Validate temperature
    if not 0.0 <= temperature <= 1.0:
        raise ValueError(f"Temperature must be between 0.0 and 1.0, got {temperature}")
    
    try:
        if provider == "OpenAI":
            return ChatOpenAI(
                model=model_name, 
                temperature=temperature, 
                api_key=SecretStr(api_key)
            )
        elif provider == "Anthropic":
            return ChatAnthropic(
                model_name=model_name, 
                temperature=temperature, 
                api_key=SecretStr(api_key),
                timeout=30.0,
                stop=None
            )
        elif provider == "OpenRouter":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=SecretStr(api_key),
                base_url=config.get("base_url"),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except Exception as e:
        raise ValueError(f"Failed to instantiate {provider} model '{model_name}': {str(e)}")

def get_models_by_research_capability(min_score: float = 8.0) -> Dict[str, Dict[str, Any]]:
    """Get models filtered by research capability score"""
    return {
        model_id: config for model_id, config in AVAILABLE_MODELS_CONFIG.items()
        if config.get("research_score", 0) >= min_score
    }

def get_models_by_reasoning_capability(min_score: float = 8.0) -> Dict[str, Dict[str, Any]]:
    """Get models filtered by reasoning capability score"""
    return {
        model_id: config for model_id, config in AVAILABLE_MODELS_CONFIG.items()
        if config.get("reasoning_score", 0) >= min_score
    }

def get_models_by_category(category: str | None = None) -> Dict[str, Dict[str, Any]]:
    """Get models filtered by category (Research Premium, Research Balanced, etc.)"""
    if not category:
        return AVAILABLE_MODELS_CONFIG
    
    return {
        model_id: config for model_id, config in AVAILABLE_MODELS_CONFIG.items()
        if category.lower() in config.get("category", "").lower()
    }

def get_models_by_provider(provider: str) -> Dict[str, Dict[str, Any]]:
    """Get models filtered by provider (OpenAI, Anthropic, OpenRouter)"""
    return {
        model_id: config for model_id, config in AVAILABLE_MODELS_CONFIG.items()
        if config.get("provider", "").lower() == provider.lower()
    }

def list_available_models() -> Dict[str, Any]:
    """Return a structured list of all available models optimized for research and reasoning"""
    result = {
        "research_premium": get_models_by_category("Research Premium"),
        "research_balanced": get_models_by_category("Research Balanced"), 
        "open_source_premium": get_models_by_category("Research Premium Open Source"),
        "open_source_balanced": get_models_by_category("Research Balanced Open Source"),
    }
    
    # Add capability-based filtering
    result["by_capability"] = {
        "top_reasoning": get_models_by_reasoning_capability(9.0),
        "strong_reasoning": get_models_by_reasoning_capability(8.5),
        "top_research": get_models_by_research_capability(9.0),
        "strong_research": get_models_by_research_capability(8.5),
    }
    
    # Add provider breakdown
    result["by_provider"] = {
        "openai": get_models_by_provider("OpenAI"),
        "anthropic": get_models_by_provider("Anthropic"),
        "google": {k: v for k, v in get_models_by_provider("OpenRouter").items() if "google" in k},
        "open_source": {k: v for k, v in get_models_by_provider("OpenRouter").items() if "google" not in k},
    }
    
    return result

def get_recommended_models_for_osint() -> Dict[str, str]:
    """Get recommended models specifically for OSINT research tasks"""
    return {
        "best_overall_reasoning": "gpt-4o",
        "best_overall_research": "claude-3-5-sonnet-20241022", 
        "best_balanced": "gpt-4o-mini",
        "best_open_source": "meta-llama/llama-3.1-405b-instruct",
        "best_multilingual_research": "qwen/qwen-2.5-72b-instruct",
        "best_mathematical_reasoning": "deepseek/deepseek-v2.5",
        "fastest_premium": "google/gemini-flash-1.5",
        "most_thorough": "claude-3-opus-20240229"
    }

def validate_model_for_research(model_id: str) -> Dict[str, Any]:
    """Validate if a model is suitable for deep research and reasoning tasks"""
    config = AVAILABLE_MODELS_CONFIG.get(model_id)
    if not config:
        return {"suitable": False, "reason": "Model not found"}
    
    reasoning_score = config.get("reasoning_score", 0)
    research_score = config.get("research_score", 0)
    
    # Set minimum thresholds for research suitability
    min_reasoning = 8.0
    min_research = 8.0
    
    is_suitable = reasoning_score >= min_reasoning and research_score >= min_research
    
    result = {
        "suitable": is_suitable,
        "model_id": model_id,
        "reasoning_score": reasoning_score,
        "research_score": research_score,
        "provider": config.get("provider"),
        "category": config.get("category"),
        "strengths": config.get("strengths")
    }
    
    if not is_suitable:
        issues = []
        if reasoning_score < min_reasoning:
            issues.append(f"Reasoning score {reasoning_score} below minimum {min_reasoning}")
        if research_score < min_research:
            issues.append(f"Research score {research_score} below minimum {min_research}")
        result["issues"] = issues
        result["recommendation"] = "Consider using a higher-tier model for complex research tasks"
    
    return result

# Example usage and testing functions
def print_model_summary():
    """Print a summary of all available models optimized for research and reasoning"""
    models = list_available_models()
    
    print("=== Elite LLM Models for Deep Research & Reasoning ===\n")
    
    # Show by research capability
    print("🎯 Models by Research & Reasoning Capability:\n")
    
    for category, models_in_category in models.items():
        if category in ["by_capability", "by_provider"]:
            continue
        
        if models_in_category:
            print(f"📁 {category.replace('_', ' ').title()}:")
            for model_id, config in models_in_category.items():
                reasoning_score = config.get('reasoning_score', 'N/A')
                research_score = config.get('research_score', 'N/A')
                print(f"   • {model_id}")
                print(f"     Provider: {config['provider']} | Reasoning: {reasoning_score}/10 | Research: {research_score}/10")
                print(f"     Strengths: {config['strengths']}")
                print()
    
    print("🏆 Top Performers (Reasoning Score ≥ 9.0):")
    top_reasoning = get_models_by_reasoning_capability(9.0)
    for model_id, config in top_reasoning.items():
        print(f"   • {model_id} - {config['reasoning_score']}/10")
    
    print("\n� Top Performers (Research Score ≥ 9.0):")
    top_research = get_models_by_research_capability(9.0)
    for model_id, config in top_research.items():
        print(f"   • {model_id} - {config['research_score']}/10")
    
    print("\n�🔧 Recommended Models for OSINT:")
    recommendations = get_recommended_models_for_osint()
    for use_case, model_id in recommendations.items():
        print(f"   • {use_case.replace('_', ' ').title()}: {model_id}")

def get_model_selection_guide() -> Dict[str, Any]:
    """Get a guide for selecting the right model based on requirements"""
    return {
        "for_complex_analysis": {
            "top_choice": "gpt-4o",
            "alternatives": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
            "reasoning": "Highest reasoning scores, proven track record with complex tasks"
        },
        "for_budget_conscious": {
            "top_choice": "gpt-4o-mini", 
            "alternatives": ["claude-3-sonnet-20240229", "google/gemini-flash-1.5"],
            "reasoning": "Strong capabilities at lower cost, good reasoning scores"
        },
        "for_open_source": {
            "top_choice": "meta-llama/llama-3.1-405b-instruct",
            "alternatives": ["deepseek/deepseek-v2.5", "qwen/qwen-2.5-72b-instruct"],
            "reasoning": "Best open source reasoning capabilities, comparable to proprietary models"
        },
        "for_multilingual_research": {
            "top_choice": "qwen/qwen-2.5-72b-instruct",
            "alternatives": ["google/gemini-pro-1.5", "meta-llama/llama-3.1-70b-instruct"],
            "reasoning": "Strong multilingual capabilities, cultural context understanding"
        }
    }

if __name__ == "__main__":
    print_model_summary()