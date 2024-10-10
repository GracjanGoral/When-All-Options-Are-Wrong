from typing import Dict, List

SUPPORTED_MODELS: Dict[str, List[str]] = {
    "OpenAI": ["gpt-4", "gpt-4o", "gpt-4-turbo"],
    "Anthropic": [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "Google": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "Replicate": [
        "meta/meta-llama-3.1-405b-instruct",
        "deepseek-ai/deepseek-math-7b-base",
        "mistralai/mistral-7b-v0.1",
    ],
    "NVIDIA": ["nvidia/nemotron-4-340b-instruct"],
    "Meta": ["meta-llama/Meta-Llama-3.1-405B-Instruct"],
}


MAPPER_MODELS: Dict[str, List[str]] = {
    "OpenAI": ["gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o-2024-08-06"],
    "Manually": ["manually"],
}
