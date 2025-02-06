"""
smart_mem_cache.py

This module implements an intelligent caching and categorization solution for LLM queries.
It uses Mem0AI as the underlying memory store and Hugging Face's zero-shot classification
to automatically assign categories to queries. The solution checks a local cache (with TTL),
searches Mem0 for near-duplicate queries, and—if needed—builds a context from similar past
interactions to compose a prompt for the LLM. It then stores the new answer in both Mem0 and
the local cache.

Additionally, it supports inbuilt LLM usage by allowing the user to choose an LLM from a set of
options ("openai", "anthropic", or "custom"). For "openai" and "anthropic", the corresponding API
keys must be set in the environment (i.e. OPENAI_API_KEY or ANTHROPIC_API_KEY). For "custom", the
user must provide a callable. A simplified debug mechanism is provided. In addition, get_answer()
can return a source flag if requested (via return_debug=True).
"""

import time
import hashlib
import os
from typing import Optional, Dict, Any, List, Tuple, Callable

from mem0 import Memory
from transformers import pipeline
from .llm_caller import *

###############################################
# Global Zero-Shot Classifier Configuration
###############################################
CANDIDATE_LABELS = [
    "personal", "sports", "coding", "shopping", "math",
    "news", "career", "health", "general"
]
zero_shot_clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def intelligent_categorize(text: str) -> str:
    if not text.strip():
        return "uncategorized"
    result = zero_shot_clf(text, CANDIDATE_LABELS)
    best_label = result["labels"][0]
    best_score = result["scores"][0]
    if best_score < 0.3:
        return "uncategorized"
    return best_label

###############################################
# Helper Functions
###############################################
def normalize_text(text: str) -> str:
    return text.strip().lower()

def make_cache_key(user_id: str, query: str) -> str:
    norm = normalize_text(query)
    raw_key = f"{user_id}::{norm}"
    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

def approximate_tokens(text: str) -> int:
    return len(text) // 4 + 1

def expand_query(text: str) -> List[str]:
    # For simplicity, we return just the original text.
    return [text]


###############################################
# Main Class: SmartCache
###############################################
class SmartCache:
    def __init__(
        self,
        memory: Optional[Memory] = None,
        similarity_threshold_reuse: float = 0.75,
        similarity_threshold_context: float = 0.4,
        max_context_tokens: int = 4096,
        ttl_seconds: Optional[int] = 1800,  # Default TTL is 30 minutes.
        debug: bool = False,
        llm_name: str = "openai",  # Options: "openai", "anthropic", "custom"
        custom_llm_callable: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize the SmartCache.
        
        :param llm_name: Which LLM to use ("openai", "anthropic", or "custom").
        :param custom_llm_callable: If llm_name is "custom", this callable will be used.
        For "openai" or "anthropic", the corresponding API key must be set in the environment.
        """
        self.memory = memory or Memory()
        self.sim_threshold_reuse = similarity_threshold_reuse
        self.sim_threshold_context = similarity_threshold_context
        self.max_context_tokens = max_context_tokens
        self.ttl_seconds = ttl_seconds
        self.cached_responses: Dict[str, Dict[str, Any]] = {}
        self.debug = debug

        # Select the LLM callable based on llm_name.
        self.llm_callable = get_llm_callable(llm_name, custom_llm_callable)

    def get_answer(
        self,
        user_id: str,
        user_query: str,
        force_refresh: bool = False,
        return_debug: bool = False
    ) -> Tuple[str, Optional[str]]:
        """
        Retrieve an answer for the user query.
        
        Workflow:
          1. Check local cache.
          2. If not found, check Mem0 for a near-duplicate.
          3. If still not found (or force_refresh is True), build context and call the LLM.
          4. Store the new answer.
          
        :param return_debug: If True, returns a tuple (answer, source) where source is one of:
                             "Local Cache", "Mem0 Near-Duplicate", or "LLM Call".
        """
        source = None
        cache_key = make_cache_key(user_id, user_query)
        
        # 1. Local Cache Check
        if not force_refresh:
            cached_answer = self.check_local_cache(cache_key)
            if cached_answer is not None:
                source = "Local Cache"
                if self.debug:
                    print(f"DEBUG: Local cache hit for key {cache_key}.")
                return (cached_answer, source) if return_debug else (cached_answer, None)
        
        # 2. Check Mem0 for a near-duplicate answer
        if not force_refresh:
            near_dup_answer = self.check_mem0_near_duplicate(user_id, user_query)
            if near_dup_answer is not None:
                source = "Mem0 Near-Duplicate"
                if self.debug:
                    print(f"DEBUG: Near-duplicate found in Mem0 for query '{user_query}'.")
                self.update_local_cache(cache_key, near_dup_answer)
                return (near_dup_answer, source) if return_debug else (near_dup_answer, None)
        
        # 3. Build Context and call LLM
        source = "LLM Call"
        context_block = self.build_context(user_id, user_query)
        prompt = self.compose_prompt(user_query, context_block)
        answer = self.llm_callable(prompt)
        if self.debug:
            print(f"DEBUG: LLM called for query '{user_query}'.")
        self.store_interaction_auto_cat(user_id, user_query, answer)
        self.update_local_cache(cache_key, answer)
        return (answer, source) if return_debug else (answer, None)

    def check_local_cache(self, cache_key: str) -> Optional[str]:
        entry = self.cached_responses.get(cache_key)
        if not entry:
            return None
        if self.ttl_seconds is not None and (time.time() - entry["timestamp"] > self.ttl_seconds):
            del self.cached_responses[cache_key]
            return None
        return entry["answer"]

    def update_local_cache(self, cache_key: str, answer: str):
        self.cached_responses[cache_key] = {"answer": answer, "timestamp": time.time()}

    def check_mem0_near_duplicate(self, user_id: str, new_query: str) -> Optional[str]:
        expansions = expand_query(new_query)
        best_answer = None
        best_score = 0.0

        for variant in expansions:
            hits = self.memory.search(query=variant, user_id=user_id)
            hits = hits[:1]  # Use only the top result.
            if not hits:
                continue
            top = hits[0]
            if self.debug:
                print(f"DEBUG: For variant '{variant}', found hit with score {top['score']} and memory: {top['memory']}")
            if top["score"] > best_score:
                best_score = top["score"]
                best_answer = top["memory"]

        if best_answer and best_score >= self.sim_threshold_reuse:
            if self.debug:
                print(f"DEBUG: Near-duplicate score {best_score} exceeds threshold {self.sim_threshold_reuse}.")
            return best_answer
        return None

    def build_context(self, user_id: str, new_query: str, top_k: int = 5) -> str:
        query_category = intelligent_categorize(new_query)
        if self.debug:
            print(f"DEBUG: Query '{new_query}' categorized as '{query_category}'.")
        expansions = expand_query(new_query)
        all_hits = []

        for variant in expansions:
            hits = self.memory.search(query=variant, user_id=user_id)
            hits = hits[:top_k]
            for h in hits:
                # Safely skip None or non-dict results
                if not h or not isinstance(h, dict):
                    continue

                # 'metadata' might be missing or None, so handle that
                cat = (h.get("metadata") or {}).get("category", "none")
                base_score = h.get("score", 0.0)
                boosted_score = base_score + (0.15 if cat == query_category else 0.0)
                all_hits.append((boosted_score, h))

        all_hits.sort(key=lambda x: x[0], reverse=True)
        relevant = [r for (sc, r) in all_hits if sc >= self.sim_threshold_context]
        if self.debug:
            print(f"DEBUG: Built context from {len(relevant)} memory entries.")

        context_snippets = []
        tokens_used = 0
        for record in relevant:
            snippet = record["memory"]  # We assume 'memory' is a string
            snippet_tokens = approximate_tokens(snippet)
            if tokens_used + snippet_tokens > self.max_context_tokens:
                break
            context_snippets.append(snippet)
            tokens_used += snippet_tokens

        return "\n\n".join(context_snippets)

    def compose_prompt(self, user_query: str, context: str) -> str:
        return f"Relevant Memory:\n{context}\n\nUser Query: {user_query}\nAssistant:"

    def store_interaction_auto_cat(self, user_id: str, user_query: str, assistant_answer: str):
        """
        Automatically categorizes the user query using zero-shot classification, then stores the
        Q&A pair in Mem0 with the category as metadata. Following the GitHub suggestion, after adding
        the memory, we update it with a string payload to avoid a 'NoneType' payload error.
        """
        category = intelligent_categorize(user_query)
        messages = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_answer},
        ]
        payload_str = f"User Query: {user_query}\nAssistant Answer: {assistant_answer}"
        metadata = {"category": category}
        try:
            result = self.memory.add(messages, user_id=user_id, metadata=metadata)
            if result and len(result) > 0:
                mem_id = result[0]['id']
                self.memory.update(memory_id=mem_id, data=payload_str)
        except Exception as e:
            if self.debug:
                print(f"DEBUG: Error storing interaction: {e}")
        if self.debug:
            print(f"DEBUG: Stored interaction for query '{user_query}' with category '{category}'.")

    def user_feedback(self, user_id: str, query: str, helpful: bool):
        if not helpful:
            cache_key = make_cache_key(user_id, query)
            if cache_key in self.cached_responses:
                del self.cached_responses[cache_key]
                if self.debug:
                    print(f"DEBUG: Removed cached answer for query '{query}' due to negative feedback.")

    def reset_all_memory(self):
        self.memory.reset()
        self.cached_responses.clear()
        if self.debug:
            print("DEBUG: Reset all Mem0 storage and local cache.")
