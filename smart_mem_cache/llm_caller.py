import os
import sys
from openai import OpenAI
from typing import *
###############################################
# LLM Callables and Router
###############################################
def openai_llm_caller(prompt: str) -> str:
    """
    Calls OpenAI's GPT-4 model and returns the response.
    Expects OPENAI_API_KEY to be set or configured in the OpenAI package.
    """
    from openai import OpenAI
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def anthropic_llm_caller(prompt: str) -> str:
    """
    A placeholder for Anthropic's LLM call.
    Replace with actual API calls if needed.
    """
    return f"[Anthropic LLM response for prompt: {prompt[:50]}...]"

def get_llm_callable(llm_name: str, custom_llm_callable: Optional[Callable[[str], str]] = None) -> Callable[[str], str]:
    """
    Returns an LLM callable based on the provided llm_name.
    Valid options are: "openai", "anthropic", or "custom".
    If "custom" is selected, custom_llm_callable must be provided.
    """
    if llm_name.lower() == "openai":
        return openai_llm_caller
    elif llm_name.lower() == "anthropic":
        return anthropic_llm_caller
    elif llm_name.lower() == "custom":
        if custom_llm_callable is None:
            raise ValueError("For llm_name 'custom', you must provide a custom_llm_callable.")
        return custom_llm_callable
    else:
        raise ValueError(f"Unknown llm_name: {llm_name}. Must be one of ['openai', 'anthropic', 'custom'].")

