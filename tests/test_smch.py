from smart_mem_cache import SmartCache

def main():
    # Initialize the cache system with debug enabled and specify using OpenAI.
    # Since we're using OpenAI, the module will check that OPENAI_API_KEY is set.
    smart_cache = SmartCache(
        similarity_threshold_reuse=0.75,
        similarity_threshold_context=0.4,
        max_context_tokens=256,
        ttl_seconds=60,   # 1-minute TTL for demonstration.
        debug=True,
        llm_name="openai"  # This tells SmartCache to use the default OpenAI LLM caller.
    )
    user_id = "alice"

    print("\n=== Use Case 1: Store an initial interaction (Exact Query) ===")
    initial_query = "Hi, I'm Alex. I like to play cricket on weekends."
    initial_answer = "Hello Alex! Great to know you enjoy cricket. I'll keep that in mind."
    smart_cache.store_interaction_auto_cat(user_id, initial_query, initial_answer)
    second_query = "I also like to workout and cook on the weekends"
    second_answer = "That's wonderful to have hobbies."
    smart_cache.store_interaction_auto_cat(user_id, second_query, second_answer)
    print("Stored initial 2 interactions.\n")

    print("=== Use Case 2: Ask a new question (non-exact match) ===")
    new_question = "What does Alex do for fun on Saturday?"
    answer2, source2 = smart_cache.get_answer(user_id, new_question, return_debug=True)
    print(f"Question: {new_question}")
    print(f"Answer: {answer2}")
    print(f"Source: {source2}\n")

    print("=== Use Case 3: Ask an exact match question (exact same query) ===")
    exact_question = "Does Alex likes to play cricket on weekend?"
    # First call: this should trigger an LLM call if not already cached.
    answer3a, source3a = smart_cache.get_answer(user_id, exact_question, return_debug=True)
    print("First call:")
    print(f"Question: {exact_question}")
    print(f"Answer: {answer3a}")
    print(f"Source: {source3a}")
    # Second call: now the exact query is expected to be in the local cache.
    answer3b, source3b = smart_cache.get_answer(user_id, exact_question, return_debug=True)
    print("\nSecond call (expected from cache):")
    print(f"Question: {exact_question}")
    print(f"Answer: {answer3b}")
    print(f"Source: {source3b}\n")

if __name__ == "__main__":
    main()

    '''
    print("=== Use Case 4: Force refresh (LLM call forced) ===")
    forced_question = exact_question  # using the same query
    answer4, source4 = smart_cache.get_answer(user_id, forced_question, default_openai_llm_caller, force_refresh=True, return_debug=True)
    print(f"Question: {forced_question}")
    print(f"Answer: {answer4}")
    print(f"Source: {source4}\n")

    print("=== Use Case 5: Negative feedback removes cached answer ===")
    smart_cache.user_feedback(user_id, exact_question, helpful=False)
    print("Negative feedback provided for the exact query.")
    # Calling the exact query again should now trigger a fresh LLM call.
    answer5, source5 = smart_cache.get_answer(user_id, exact_question, default_openai_llm_caller, return_debug=True)
    print(f"Question: {exact_question}")
    print(f"Answer: {answer5}")
    print(f"Source: {source5}\n")
    '''
