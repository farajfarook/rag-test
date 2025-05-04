import torch


def retrieve_with_ai_optimized_query(prompt, tokenizer, model, vector_store):
    # 1. Create a prompt to ask the model for relevant search terms
    search_term_prompt = (
        f"Based on the following question, identify the key entities, concepts, or keywords that would be most effective for searching a database containing candidate profiles and job descriptions. Return only the search terms, separated by commas.\n\n"
        f"Question: {prompt}\n\n"
        f"Search Terms:"
    )

    # 2. Use the model to generate search terms
    # Encode the prompt
    input_ids = tokenizer.encode(search_term_prompt, return_tensors="pt").to(
        model.device
    )

    # Generate the response (search terms)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=50,  # Limit token generation for search terms
            temperature=0.2,  # Lower temperature for more focused output
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract search terms (remove the original prompt part)
    search_terms_str = generated_text.replace(search_term_prompt, "").strip()
    print(f"Generated search terms string: {search_terms_str}")

    # Simple parsing: split by comma, strip whitespace
    search_terms = [
        term.strip() for term in search_terms_str.split(",") if term.strip()
    ]

    if not search_terms:
        print("Model did not generate specific search terms, using original prompt.")
        # Fallback to original prompt if no terms generated
        search_query = prompt
    else:
        # Combine terms into a single query string for the vector store
        search_query = " ".join(search_terms)
        print(f"Using combined search query: {search_query}")

    # 3. Search the vector store using the generated/refined query
    retrieved_contexts = vector_store.search(search_query)
    return retrieved_contexts
