# Smart Cache Package

**Smart Cache Package** is a pip‑installable Python library that provides an intelligent caching and LLM routing solution. It leverages:
- **Mem0AI** for storing and retrieving past interactions.
- **Hugging Face's zero-shot classification** to automatically assign categories to queries.
- Built‑in integration with popular LLM providers (e.g. OpenAI, Anthropic) and support for custom LLM callables.

## Features

- **Caching and Reuse:**  
  Caches previous query–answer pairs to avoid unnecessary LLM calls.
  
- **Near‑Duplicate Search:**  
  Searches Mem0 for near‑duplicate queries. If a query similar enough is found (based on a configurable threshold), it is returned from cache or Mem0 without calling the LLM.

- **Context Building:**  
  If no exact or near‑duplicate answer exists, the package builds a context from related stored interactions to provide richer prompts for the LLM.

- **Built-in LLM Routing:**  
  Supports multiple LLM providers:
  - **OpenAI:** Uses `OPENAI_API_KEY` from the environment.
  - **Anthropic:** Uses `ANTHROPIC_API_KEY` (placeholder; extend as needed).
  - **Custom:** Allows users to provide their own LLM callable.
  
- **Debugging and Source Reporting:**  
  When desired, `get_answer()` can return an extra flag indicating the source of the answer (e.g. "Local Cache", "Mem0 Near-Duplicate", "LLM Call").

## Installation

You can install the package via pip (once published):

```bash
pip install smart_cache_package

