from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI()

# Changed to a smaller Falcon model for better speed
model_name = "tiiuae/falcon3-1b-instruct"
tokenizer_name = "tiiuae/falcon3-1b-instruct"


# Set up Chroma vector database
class ChromaVectorStore:
    def __init__(self, collection_name="candidate_docs"):
        # Create a persistent client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Set up the embedding function to use our sentence transformer model
        self.sentence_transformer_ef = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

        # Create or get the collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name, embedding_function=self.sentence_transformer_ef
            )
            print(f"Using existing collection: {collection_name}")
        except Exception:  # Collection doesn't exist
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.sentence_transformer_ef,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Created new collection: {collection_name}")

    def add_document(self, text, candidate_name):
        """Add a single document with candidate name prefixed"""
        if not text:
            return

        # Prefix the document with candidate name for context
        full_doc = f"[Candidate: {candidate_name}] {text}"

        # Generate a document ID
        doc_id = f"doc_{len(self.collection.get()['ids']) + 1}"

        # Add document to Chroma
        self.collection.add(
            documents=[full_doc],
            ids=[doc_id],
            metadatas=[{"candidate": candidate_name}],
        )

    def search(self, query, top_k=3):
        # If no documents, return empty list
        if len(self.collection.get()["ids"]) == 0:
            return []

        # Query the collection
        results = self.collection.query(query_texts=[query], n_results=top_k)

        # Return the documents
        return results["documents"][0] if results["documents"] else []


# Initialize vector store
vector_store = ChromaVectorStore()

try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Loading model with lower precision and device mapping for better memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,  # Required for some models including Falcon
    )
    print(f"Model and tokenizer loaded successfully from {model_name}.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()


def generate_response(prompt, retrieved_contexts=None):
    # Format prompt with retrieved contexts for RAG
    if retrieved_contexts:
        context_text = "\n".join(retrieved_contexts)
        formatted_prompt = f"User: I have the following information:\n{context_text}\n\nWith this context, {prompt}\nAssistant: "
    else:
        formatted_prompt = f"User: {prompt}\nAssistant: "

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


class PromptInput(BaseModel):
    prompt: str
    use_rag: bool = True


class DocumentInput(BaseModel):
    texts: List[str]
    candidate_name: str  # Required field for candidate name


class ResponseOutput(BaseModel):
    response: str
    retrieved_contexts: Optional[List[str]] = None


@app.post("/generate")
async def generate(prompt_input: PromptInput):
    try:
        if prompt_input.use_rag:
            retrieved_contexts = vector_store.search(prompt_input.prompt)
            response = generate_response(prompt_input.prompt, retrieved_contexts)
            return {"response": response, "retrieved_contexts": retrieved_contexts}
        else:
            response = generate_response(prompt_input.prompt)
            return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_documents")
async def add_documents(document_input: DocumentInput):
    try:
        # Merge all texts into a single document with proper formatting
        merged_text = "\n\n".join(document_input.texts)

        # Add as a single document with candidate name
        vector_store.add_document(merged_text, document_input.candidate_name)

        return {
            "status": "success",
            "document_count": len(vector_store.collection.get()["ids"]),
            "candidate": document_input.candidate_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
