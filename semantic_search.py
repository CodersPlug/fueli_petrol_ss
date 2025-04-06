import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("Error: OPENAI_API_KEY not found in .env file")
    exit(1)

print("Initializing OpenAI client...")
client = OpenAI(api_key=api_key)

# Create storage directory if it doesn't exist
storage_dir = Path("qdrant_storage")
storage_dir.mkdir(exist_ok=True)

# Initialize Qdrant client in local mode
print("Initializing Qdrant client...")
qdrant_client = QdrantClient(path=str(storage_dir))

# Collection configuration
COLLECTION_NAME = "petrol_transactions"
VECTOR_SIZE = 3072

def get_embedding(text: str, model="text-embedding-3-large") -> List[float]:
    """Get embeddings for a text using OpenAI's API."""
    try:
        print(f"Getting embedding for text: {text[:50]}...")
        response = client.embeddings.create(
            model=model,
            input=text,
            dimensions=VECTOR_SIZE
        )
        print("Successfully got embedding")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def semantic_search(query: str, top_k: int = 5) -> List[Dict]:
    """Search for most similar records using Qdrant."""
    print(f"\nProcessing search query: {query}")
    
    # Get query embedding
    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Failed to get query embedding")
        return []
    
    # Search in Qdrant
    print("Searching for similar records...")
    try:
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Format results
        results = [
            {
                "text": hit.payload["text"],
                "similarity": hit.score,
                "created_at": hit.payload["created_at"]
            }
            for hit in search_results
        ]
        print(f"Found {len(results)} results")
        return results
    except Exception as e:
        print(f"Error searching in Qdrant: {e}")
        return []

def analyze_with_gpt4(query: str, relevant_data: List[Dict]) -> str:
    """Use GPT-4 to analyze the retrieved data and provide an intelligent answer."""
    try:
        # Prepare the context for GPT-4
        context = "\n".join([f"- {data['text']}" for data in relevant_data])
        
        # Create the system message
        system_message = """Eres un experto analista de datos de ventas de combustible. 
        Tu tarea es analizar los datos proporcionados y responder preguntas sobre las ventas.
        Responde de manera CONCISA y DIRECTA, sin explicaciones adicionales.
        Si la pregunta requiere análisis de totales o tendencias, calcula los números exactos.
        Responde siempre en español y en una sola línea."""
        
        # Create the user message
        user_message = f"""Pregunta: {query}

Datos relevantes:
{context}

Responde de manera concisa y directa:"""
        
        # Call GPT-4
        print("Analizando datos con GPT-4...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,  # Reduced for more deterministic answers
            max_tokens=150    # Reduced for shorter answers
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error al analizar con GPT-4: {e}")
        return "Lo siento, hubo un error al procesar tu pregunta."

def main():
    try:
        # Verify connection to Qdrant
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            print(f"Connected to Qdrant. Collection contains {collection_info.points_count} points")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            print("Make sure data is imported using import_to_qdrant.py")
            return
        
        print("\n=== Sistema de Búsqueda Semántica Inteligente ===")
        print("Escribe tus preguntas en español sobre las transacciones de combustible.")
        print("Ejemplos:")
        print("- ¿Cuál es el producto más vendido?")
        print("- ¿Cuántas ventas hubo en enero?")
        print("- ¿Cuál fue la venta más grande?")
        print("Escribe 'quit' para salir")
        print("=" * 50)
        
        while True:
            try:
                query = input("\nTu pregunta: ").strip()
                if not query:
                    continue
                    
                if query.lower() == 'quit':
                    print("Saliendo del programa...")
                    break
                
                # Get relevant data
                results = semantic_search(query, top_k=10)  # Increased to get more context
                if not results:
                    print("No se encontraron resultados relevantes.")
                    continue
                
                # Analyze with GPT-4
                answer = analyze_with_gpt4(query, results)
                print("\nRespuesta:")
                print(answer)
                
                # Show supporting data
                print("\nDatos de soporte:")
                for i, result in enumerate(results[:3], 1):  # Show top 3 supporting records
                    print(f"\n{i}. Similitud: {result['similarity']:.4f}")
                    print(f"   {result['text']}")

            except KeyboardInterrupt:
                print("\nSaliendo del programa...")
                break
            except Exception as e:
                print(f"Error procesando la consulta: {e}")

    except Exception as e:
        print(f"Error general: {e}")

if __name__ == "__main__":
    main() 