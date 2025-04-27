import json
from pathlib import Path
import logging
import time
# Import SentenceTransformer library
from sentence_transformers import SentenceTransformer
import numpy as np # For handling embeddings as arrays if needed

# --- Configuration ---
INPUT_JSON = Path("corpus_chunks.json") # Input file from text_chunker_v1
OUTPUT_JSON = Path("corpus_with_embeddings.json") # Output file
# Choose a model from Sentence Transformers library:
# Common choices: 'all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1', 'BAAI/bge-small-en-v1.5'
MODEL_NAME = 'all-MiniLM-L6-v2'
# Set batch size for embedding generation (adjust based on your RAM/GPU memory)
BATCH_SIZE = 32

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Function ---

def generate_embeddings():
    """Loads chunked text, generates embeddings, and saves the combined data."""

    # --- 1. Load Chunked Data ---
    if not INPUT_JSON.exists():
        logging.error(f"Input file not found: {INPUT_JSON}")
        logging.error("Please run the chunking script (e.g., chunk_corpus.py) first.")
        return

    logging.info(f"Loading chunked data from {INPUT_JSON}...")
    try:
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        logging.info(f"Loaded {len(chunks_data)} chunks.")
        if not chunks_data:
            logging.warning("Input file contains no chunks. Exiting.")
            return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {INPUT_JSON}: {e}")
        return
    except Exception as e:
        logging.error(f"Error loading {INPUT_JSON}: {e}")
        return

    # Extract just the text content for batch processing
    texts_to_embed = [chunk['text'] for chunk in chunks_data]

    # --- 2. Load Embedding Model ---
    logging.info(f"Loading Sentence Transformer model: '{MODEL_NAME}'...")
    try:
        # This will download the model on the first run if not already cached locally
        # You can specify device='cuda' if you have a compatible GPU and PyTorch installed
        model = SentenceTransformer(MODEL_NAME, device='cpu') # Use 'cuda' for GPU
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model '{MODEL_NAME}': {e}")
        logging.error("Ensure 'sentence-transformers' and 'torch' (or 'tensorflow') are installed.")
        return

    # --- 3. Generate Embeddings ---
    logging.info(f"Generating embeddings for {len(texts_to_embed)} text chunks (batch size: {BATCH_SIZE})...")
    start_time = time.time()
    try:
        # Use model.encode() which handles batching efficiently
        embeddings = model.encode(
            texts_to_embed,
            batch_size=BATCH_SIZE,
            show_progress_bar=True # Display a progress bar in the console
        )
        # Embeddings will be a numpy array where each row is an embedding vector
        logging.info(f"Embeddings generated. Shape: {embeddings.shape}")

    except Exception as e:
        logging.error(f"Error during embedding generation: {e}", exc_info=True)
        return

    end_time = time.time()
    logging.info(f"Embedding generation took {end_time - start_time:.2f} seconds.")

    # --- 4. Combine Chunks with Embeddings ---
    logging.info("Combining original chunk data with embeddings...")
    corpus_with_embeddings = []
    if len(chunks_data) != len(embeddings):
         logging.error("Mismatch between number of chunks and generated embeddings. Aborting.")
         return

    for i, chunk in enumerate(chunks_data):
        # Convert numpy array row to a standard Python list for JSON serialization
        chunk['embedding'] = embeddings[i].tolist()
        corpus_with_embeddings.append(chunk)

    # --- 5. Save Output ---
    logging.info(f"Saving data with embeddings to {OUTPUT_JSON.resolve()}...")
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(corpus_with_embeddings, f, ensure_ascii=False, indent=2) # Use indent for readability
        logging.info("Output saved successfully.")
    except IOError as e:
        logging.error(f"Failed to save output to {OUTPUT_JSON}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during saving: {e}")


if __name__ == "__main__":
    generate_embeddings()
