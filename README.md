# RAG-Based Question Answering API

A FastAPI-based API using Retrieval-Augmented Generation (RAG) to provide answers based on candidate data.

## Features

- FastAPI backend with Falcon 3-1B Instruct model
- Sentence Transformer embeddings for semantic search
- Configurable RAG (can be turned on/off per request)

## Quick Start

```
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

## API Endpoints

- POST `/generate`: Generate responses with or without RAG
- POST `/add_documents`: Add documents to the vector store

## Project Structure

```
main.py                # Main FastAPI application
test.http              # HTTP request examples
candidate-data/        # Candidate data files
chroma_db/             # Vector database storage
```
