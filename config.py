import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

# Three model roles — kept as separate vars so each can be tuned independently,
# but all default to gpt-4.1 for reliable structured tool calling:
#
#   SYNTHESIS_MODEL      — final user-facing answer + tool calling.
#   EXTRACTION_MODEL     — batch FSR extraction at ingestion time (plain JSON).
#   CLASSIFICATION_MODEL — 4-way intent pre-classification, returns a single word.
#
SYNTHESIS_MODEL        = os.getenv("SYNTHESIS_MODEL",        "gpt-4.1")
EXTRACTION_MODEL       = os.getenv("EXTRACTION_MODEL",       "gpt-4.1")
CLASSIFICATION_MODEL   = os.getenv("CLASSIFICATION_MODEL",   "gpt-4.1")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

CHROMA_DB_PATH      = os.getenv("CHROMA_DB_PATH",      "./processed/chroma_db")
SERVICE_DB_PATH     = os.getenv("SERVICE_DB_PATH",      "./processed/service_records.db")
DATA_BROCHURES_DIR  = os.getenv("DATA_BROCHURES_DIR",   "./data/Products")
DATA_SERVICES_DIR   = os.getenv("DATA_SERVICES_DIR",    "./data/HistoricalServices")
DATA_FLOTATION_CSV  = os.getenv("DATA_FLOTATION_CSV",   "./data/flotation_process_data.csv")

MEMORY_COMPRESS_AFTER_TURNS = int(os.getenv("MEMORY_COMPRESS_AFTER_TURNS", "6"))

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
