import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vector_store import ChromaVectorStore

app = FastAPI()
vector_store = ChromaVectorStore()

model_name = "tiiuae/falcon3-1b-instruct"
tokenizer_name = "tiiuae/falcon3-1b-instruct"

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
    print("Model and tokenizer loaded successfully from {model_name}.")
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
    print("Candidate data loaded and added to vector store.")
except Exception as e:
    print(f"An error occurred while loading candidate data: {e}")
    exit()


def retrieve(prompt):
    retrieved_contexts = vector_store.search(prompt)
    return retrieved_contexts


def augment(retrieved_contexts, prompt):
    context_text = "\n".join(retrieved_contexts)
    formatted_prompt = f"User: I have the following information:\n{context_text}\n\nWith this context, {prompt}\nAssistant: "
    return formatted_prompt


def generate(formatted_prompt):
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


@app.post("/generate")
async def generate(prompt: str):
    try:
        # RETRIEVE Context from Vector Store
        retrieved_contexts = retrieve(prompt)
        # AUGMENT the prompt with retrieved contexts
        formatted_prompt = augment(retrieved_contexts, prompt)
        # GENERATE the response using the model
        response = generate(formatted_prompt)
        return {"response": response, "retrieved_contexts": retrieved_contexts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
