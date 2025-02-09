# Smart Cache Package

**Smart Cache Package** is a pip-installable Python library that provides an **intelligent caching and LLM routing solution**. It efficiently reduces redundant LLM calls while ensuring that queries benefit from relevant past interactions.

This package leverages:
- **Mem0AI** for storing and retrieving past interactions.
- **Hugging Face's zero-shot classification** to automatically assign categories to queries.
- **Built-in LLM integrations** for OpenAI (default), Anthropic (coming soon), and support for custom LLM callables.

---

## 🚀 Features

### ✅ **Smart Caching and Query Reuse**
- Caches previous **query–answer** pairs to avoid unnecessary LLM calls.
- Uses Mem0AI to store user interactions for future reference.

### 🔍 **Near-Duplicate Detection**
- Searches Mem0 for **semantically similar** queries.
- If a near-duplicate query is found (above a configurable similarity threshold), it retrieves the stored response.

### 📄 **Context Building for Richer Responses**
- If no exact or near-duplicate answer exists, it **retrieves related stored interactions** to provide context to the LLM.
- Ensures the **LLM receives additional context**, improving coherence without exceeding token limits.

### 🔄 **Built-in LLM Routing**
- **Supports multiple LLM providers:**
  - **OpenAI (Default):** Uses `OPENAI_API_KEY` from environment variables.
  - **Custom LLM:** Users can provide their own callable function.
 
### 🔄 **Export & Import Cache Data (New Feature)**
- Users can **export** their cache to a file and **import** it into another instance of SmartCache.
- This feature enables **collaboration and 

### 🛠 **Debugging and Transparency**
- `get_answer()` can return an **extra debug flag** indicating the **source of the answer**:
  - `"Local Cache"` → Retrieved from in-memory cache.
  - `"Mem0 Near-Duplicate"` → Retrieved from Mem0.
  - `"LLM Call"` → Fetched from an external LLM.

---

## 📥 Installation


```bash
pip install smart_cache_package
```


## PREREQUISITES 
- Make sure you export OPENAI_API_KEY even if you use a custom LLM, it is required for storing the embeddings using the embedding model from openai.
- Custom Embedding Model support coming soon.
```
export OPENAI_API_KEY = <your_openai_key>
```


## 🚀 Usage

### **1️⃣ Initialize SmartCache**
The package **automatically picks OpenAI** as the default LLM **if no custom callable is provided**.

```python
from smart_mem_cache import SmartCache

# Initialize the cache system (requires OPENAI_API_KEY)
smart_cache = SmartCache(
    similarity_threshold_reuse=0.75,
    similarity_threshold_context=0.4,
    max_context_tokens=256,
    ttl_seconds=60,  # 1-minute TTL
    debug=True,
    llm_name="openai"  # Use OpenAI by default
)

user_id = "alice"

```

### Store Initial Interactions
Store a user’s conversation history for retrieval later.
```python
smart_cache.store_interaction_auto_cat(
    user_id, 
    "Hi, I'm Alex. I like to play cricket on weekends.", 
    "Hello Alex! Great to know you enjoy cricket. I'll keep that in mind."
)

smart_cache.store_interaction_auto_cat(
    user_id, 
    "I also like to workout and cook on the weekends", 
    "That's wonderful to have hobbies."
)

print("Stored initial interactions.")
```

### Ask a New Question
If an exact match or near-duplicate exists, it will reuse the previous answer. Otherwise, it calls the LLM.
```python
new_question = "What does Alex do for fun on Saturday?"
answer, source = smart_cache.get_answer(user_id, new_question, return_debug=True)

print(f"Question: {new_question}")
print(f"Answer: {answer}")
print(f"Source: {source}")  # Could be "Local Cache", "Mem0 Near-Duplicate", or "LLM Call"
```

### Ask an Exact Match Question
If a user repeats a previously asked query, it should return from cache.
```python
exact_question = "Does Alex like to play cricket on weekends?"

# First call (expected LLM call)
answer1, source1 = smart_cache.get_answer(user_id, exact_question, return_debug=True)
print(f"First Call - Source: {source1}")

# Second call (should return from cache)
answer2, source2 = smart_cache.get_answer(user_id, exact_question, return_debug=True)
print(f"Second Call - Source: {source2}")  # Expected: "Local Cache"
```

### Force Refresh to Bypass Cache
If you need to force a fresh answer from LLM, use force_refresh=True.

```python
forced_question = "Any weekend hobbies for Alex?"
answer, source = smart_cache.get_answer(user_id, forced_question, force_refresh=True, return_debug=True)

print(f"Forced Refresh - Source: {source}")  # Expected: "LLM Call"

```

### Provide Negative Feedback (Remove Cached Answer)
```python
smart_cache.user_feedback(user_id, exact_question, helpful=False)
print("Negative feedback provided, answer removed from cache.")

# Asking again should now trigger a fresh LLM call.
answer, source = smart_cache.get_answer(user_id, exact_question, return_debug=True)
print(f"Re-asked after negative feedback - Source: {source}")  # Expected: "LLM Call"

```

### Export & Import Cached Data (New Feature)
Exporting Cache to a JSON File:
```python
smart_cache.export_cache("cache_data.json")
print("Cache exported successfully.")
```
Importing Cache from a JSON File:
```python
smart_cache.import_cache("cache_data.json")
print("Cache imported successfully.")
```

### Using a Custom LLM (Instead of OpenAI)
If you want to use a custom LLM (e.g., a self-hosted model), provide a callable function.
```python
def custom_llm(prompt: str) -> str:
    return f"[Custom LLM response for: {prompt[:50]}...]"

# Initialize SmartCache with custom LLM
smart_cache = SmartCache(
    similarity_threshold_reuse=0.75,
    similarity_threshold_context=0.4,
    max_context_tokens=256,
    ttl_seconds=60,
    debug=True,
    llm_callable=custom_llm  # Pass custom LLM function
)
```

### Environment Variables (Required for OpenAI & Anthropic)
If you use OpenAI or Anthropic, ensure the API keys are set as environment variables.
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```
### 🚀 Enjoy smart caching and optimized LLM interactions! 🚀
