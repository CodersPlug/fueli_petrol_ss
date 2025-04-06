# Fuel Station Analytics

Sistema de análisis semántico de transacciones de combustible utilizando OpenAI Embeddings y Qdrant Vector Database.

## Características

- Búsqueda semántica de transacciones
- Análisis inteligente usando GPT-4
- Interfaz web con Streamlit
- Base de datos vectorial con Qdrant
- Procesamiento de datos en español

## Requisitos

- Python 3.11+
- OpenAI API Key
- Streamlit
- Qdrant
- pandas
- tiktoken

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/CodersPlug/fueli_petrol_ss.git
cd fueli_petrol_ss
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Crear archivo .env con la API key de OpenAI:
```bash
OPENAI_API_KEY=tu_api_key
```

## Uso

1. Procesar datos y crear embeddings:
```bash
python process_tokens.py
```

2. Importar datos a Qdrant:
```bash
python import_to_qdrant.py
```

3. Ejecutar la interfaz web:
```bash
streamlit run app.py
```

## Estructura del Proyecto

- `app.py`: Interfaz web con Streamlit
- `process_tokens.py`: Procesamiento de datos y creación de embeddings
- `import_to_qdrant.py`: Importación de datos a Qdrant
- `requirements.txt`: Dependencias del proyecto

## Licencia

MIT 