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

## Instalación Local

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

## Uso Local

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

## Despliegue en Streamlit Cloud

1. Fork este repositorio a tu cuenta de GitHub

2. Visita [share.streamlit.io](https://share.streamlit.io)

3. Conecta tu cuenta de GitHub y selecciona el repositorio

4. En la configuración del despliegue, agrega las siguientes variables secretas:
   - `OPENAI_API_KEY`: Tu API key de OpenAI

5. Haz clic en "Deploy!"

La aplicación estará disponible en una URL pública de Streamlit Cloud.

## Estructura del Proyecto

- `app.py`: Interfaz web con Streamlit
- `process_tokens.py`: Procesamiento de datos y creación de embeddings
- `import_to_qdrant.py`: Importación de datos a Qdrant
- `requirements.txt`: Dependencias del proyecto
- `.streamlit/config.toml`: Configuración de Streamlit
- `packages.txt`: Dependencias del sistema

## Licencia

MIT 