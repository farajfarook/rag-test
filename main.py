import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from retrievers import retrieve_with_ai_optimized_query
from vector_store import ChromaVectorStore

app = FastAPI()
vector_store = ChromaVectorStore()

models = {
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "meta-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "falcon-3b": "tiiuae/Falcon3-3B-Instruct",
    "dseek-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

selected_model = "dseek-7b"

model_name = models[selected_model]
tokenizer_name = models[selected_model]

# Load the model and tokenizer
try:
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    print(f"Model and tokenizer loaded successfully from {model_name}.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

# Load candidate data from JSON file
try:
    print("Loading candidate data...")
    with open("candidate_data.json", "r") as f:
        candidate_data = json.load(f)
        for candidate in candidate_data:
            candidate_name = candidate["candidate_name"]
            texts = candidate["texts"]
            merged_text = "\n\n".join(texts)
            vector_store.add_document(merged_text, candidate_name)
    count = vector_store.collection.count()
    print(f"Candidate data loaded and added to vector store. Count: {count}")
except Exception as e:
    print(f"An error occurred while loading candidate data: {e}")
    exit()


# RETRIEVE Context from Vector Store
def retrieve(prompt):
    # retrieved_contexts = vector_store.search(prompt)
    retrieved_contexts = retrieve_with_ai_optimized_query(
        prompt, tokenizer, model, vector_store
    )
    return retrieved_contexts


# AUGMENT the prompt with retrieved contexts
def augment(retrieved_contexts, prompt):
    if not retrieved_contexts:
        print("No retrieved contexts available, using default context.")
        context_text = "No relevant information found."
    else:
        context_text = "\n".join(retrieved_contexts)
    # Add instruction to the prompt template
    formatted_prompt = (
        f"User: I have the following information:\n{context_text}\n\n"
        f"With this context, answer the following question: {prompt}\n"
        f"Assistant: "
    )
    return formatted_prompt


# GENERATE the response using the model
def generate(formatted_prompt):
    print(f"Formatted prompt: {formatted_prompt}")
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response.replace(formatted_prompt, "").strip()
    return response


# Define request body model
class GenerateRequest(BaseModel):
    prompt: str


# FastAPI endpoint for generating responses
@app.post("/generate")
async def generate_api(request: GenerateRequest):
    try:
        # RETRIEVE Context using AI-optimized query
        retrieved_contexts = retrieve(request.prompt)
        # AUGMENT the prompt with retrieved contexts
        formatted_prompt = augment(retrieved_contexts, request.prompt)
        # GENERATE the response using the model
        response = generate(formatted_prompt)
        return {"response": response}
    except Exception as e:
        # Log the exception for debugging
        print(f"Error in /generate endpoint: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
