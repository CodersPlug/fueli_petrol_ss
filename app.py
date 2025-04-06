import streamlit as st

# Configure Streamlit page - must be the first st command
st.set_page_config(
    page_title="Análisis de Ventas de Combustible",
    page_icon="⛽",
    layout="wide"
)

import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = st.secrets["OPENAI_API_KEY"]
if not api_key:
    st.error("Error: OPENAI_API_KEY not found in secrets")
    st.stop()

client = OpenAI(api_key=api_key)

# Initialize Qdrant client in memory
@st.cache_resource
def get_qdrant_client():
    """Get or create a singleton Qdrant client instance in memory."""
    return QdrantClient(":memory:")  # Use in-memory storage

# Collection configuration
COLLECTION_NAME = "petrol_transactions"
VECTOR_SIZE = 3072

@st.cache_data
def load_embeddings():
    """Load embeddings from the JSON file."""
    with open('embeddings.json', 'r') as f:
        return json.load(f)

def initialize_collection(qdrant_client: QdrantClient):
    """Initialize the Qdrant collection if it doesn't exist."""
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
    except Exception:
        # Collection doesn't exist, create it
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        
        # Load embeddings and data
        embeddings_data = load_embeddings()
        
        # Prepare points for insertion
        points = []
        for i, item in enumerate(embeddings_data):
            points.append(
                models.PointStruct(
                    id=i,
                    vector=item['embedding'],
                    payload={
                        "text": item['text'],
                        "created_at": item.get('created_at', '')
                    }
                )
            )
        
        # Insert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
        
        st.success(f"Initialized collection with {len(points)} records")

# Get the Qdrant client instance
qdrant_client = get_qdrant_client()

# Initialize collection on startup
with st.spinner("Inicializando base de datos..."):
    initialize_collection(qdrant_client)

def get_embedding(text: str, model="text-embedding-3-large") -> List[float]:
    """Get embeddings for a text using OpenAI's API."""
    try:
        response = client.embeddings.create(
            model=model,
            input=text,
            dimensions=VECTOR_SIZE
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return None

def semantic_search(query: str, top_k: int = 5) -> List[Dict]:
    """Search for most similar records using Qdrant."""
    # Get query embedding
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []
    
    # Search in Qdrant
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
        return results
    except Exception as e:
        st.error(f"Error searching in Qdrant: {e}")
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
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al analizar con GPT-4: {e}")
        return "Lo siento, hubo un error al procesar tu pregunta."

def main():
    # Add title and description
    st.title("Análisis de Ventas de Combustible")
    st.markdown("""
    Este sistema permite realizar búsquedas semánticas sobre las transacciones de combustible.
    Puedes hacer preguntas en español sobre las ventas, productos, fechas y más.
    """)
    
    # Add example questions
    with st.expander("Ejemplos de preguntas"):
        st.markdown("""
        - ¿Cuál es el producto más vendido?
        - ¿Cuántas ventas hubo en enero?
        - ¿Cuál fue la venta más grande?
        - ¿En qué pico se despachó más NS XXI?
        - ¿Cuánto se facturó en total de NS XXI?
        """)
    
    # Add search input
    query = st.text_input(
        "Escribe tu pregunta:",
        placeholder="Ej: ¿Cuál es el producto más vendido?",
        help="Escribe tu pregunta en español sobre las ventas de combustible"
    )
    
    if query:
        with st.spinner("Analizando..."):
            # Get relevant data
            results = semantic_search(query, top_k=10)
            
            if not results:
                st.warning("No se encontraron resultados relevantes.")
            else:
                # Analyze with GPT-4
                answer = analyze_with_gpt4(query, results)
                
                # Display answer in a clean format
                st.markdown("### Respuesta:")
                st.info(answer)

if __name__ == "__main__":
    main() 