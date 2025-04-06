import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Create storage directory if it doesn't exist
storage_dir = Path("qdrant_storage")
storage_dir.mkdir(exist_ok=True)

# Initialize Qdrant client in local mode
qdrant_client = QdrantClient(path=str(storage_dir))

# Collection configuration
COLLECTION_NAME = "petrol_transactions"
VECTOR_SIZE = 3072

def init_collection():
    """Initialize Qdrant collection if it doesn't exist."""
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection {COLLECTION_NAME} already exists")
    except Exception:
        print(f"Creating collection {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

def load_data() -> tuple[List[Dict[str, Any]], List[List[float]]]:
    """Load both tokenized data and embeddings from JSON files."""
    try:
        print("Loading tokenized data...")
        with open("tokenized_data.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("Loading pre-computed embeddings...")
        with open("embeddings.json", 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
            
        print(f"Successfully loaded {len(data)} records and their embeddings")
        return data, embeddings
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

def store_vectors_in_qdrant(data: List[Dict], embeddings: List[List[float]], batch_size: int = 100):
    """Store pre-computed embeddings in Qdrant."""
    print(f"\nStoring {len(embeddings)} vectors in Qdrant in batches of {batch_size}")
    
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Storing vectors"):
        batch_data = data[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        # Skip None embeddings
        valid_points = [
            (idx, emb, text) for idx, (emb, text) in enumerate(zip(batch_embeddings, batch_data))
            if emb is not None
        ]
        
        if not valid_points:
            continue
            
        # Prepare points for Qdrant
        points = [
            models.PointStruct(
                id=i + idx,
                vector=emb,
                payload={
                    "text": text["text"],
                    "created_at": datetime.now().isoformat()
                }
            )
            for idx, emb, text in valid_points
        ]
        
        try:
            # Upload to Qdrant
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            print(f"\nSuccessfully stored batch {i}-{i+len(batch_data)}")
        except Exception as e:
            print(f"\nError storing batch {i}-{i+batch_size}: {e}")

def main():
    try:
        print("Connecting to Qdrant...")
        
        # Initialize Qdrant collection
        init_collection()
        
        # Check if we need to load and store vectors
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        if collection_info.points_count == 0:
            print("Collection is empty, loading pre-computed embeddings...")
            data, embeddings = load_data()
            if not data or not embeddings:
                print("No data or embeddings loaded, exiting...")
                return
                
            print("\nStoring pre-computed embeddings in Qdrant...")
            store_vectors_in_qdrant(data, embeddings)
        else:
            print(f"Collection already contains {collection_info.points_count} points")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 