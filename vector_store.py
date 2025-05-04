import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import hashlib


class ChromaVectorStore:
    def __init__(self, collection_name="candidate_docs"):
        # Create a persistent client with telemetry disabled
        self.client = chromadb.PersistentClient(
            path="./chroma_db", settings=Settings(anonymized_telemetry=False)
        )

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
        if not text:
            return

        # Prefix the document with candidate name for context
        full_doc = f"[Candidate: {candidate_name}] {text}"

        # Generate a deterministic document ID based on the candidate name
        doc_id = hashlib.sha256(candidate_name.encode()).hexdigest()

        # Add document to Chroma (upsert behavior based on candidate name)
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
