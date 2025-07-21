import os
from pathlib import Path

# Model Configuration
MODEL_NAME = "qwen2.5-coder:7b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"
CONTEXT_WINDOW = 32768
MAX_TOKENS = 4096
TEMPERATURE = 0.1

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
VECTOR_DB_PATH = DATA_DIR / "lancedb"

# Chunking Configuration
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_QUERY = 10

# Create directories
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Performance Settings
BATCH_SIZE = 50
MAX_WORKERS = 4 