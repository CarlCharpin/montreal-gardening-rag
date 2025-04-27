import os
from pathlib import Path
import json
import re
import logging
from typing import List, Dict, Tuple

# --- Configuration ---
CORPUS_DIR = Path("text-corpus")  # Directory containing your .txt files
OUTPUT_JSON = Path("corpus_chunks.json") # Output file for chunks
CHUNK_SIZE = 500  # Target size of each chunk (in characters)
CHUNK_OVERLAP = 50  # Number of characters to overlap between chunks

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Remove excessive whitespace (multiple newlines, leading/trailing spaces)
    text = re.sub(r'\n\s*\n', '\n\n', text) # Replace multiple newlines with double newline
    text = re.sub(r'[ \t]+', ' ', text)      # Replace multiple spaces/tabs with single space
    text = text.strip()                     # Remove leading/trailing whitespace
    return text

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits text into chunks of specified size with overlap."""
    if not text:
        return []

    chunks = []
    start_index = 0
    text_length = len(text)

    while start_index < text_length:
        end_index = min(start_index + chunk_size, text_length)
        chunks.append(text[start_index:end_index])
        
        # Move start index for the next chunk
        start_index += chunk_size - chunk_overlap
        
        # If overlap pushes start_index past the end, we're done
        if start_index >= end_index:
             break # Avoid infinite loop on very small texts or large overlaps

    return chunks

def load_and_chunk_corpus(corpus_dir: Path, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Loads .txt files from a directory and splits them into chunks with metadata."""
    chunked_corpus = []
    if not corpus_dir.is_dir():
        logging.error(f"Corpus directory not found: {corpus_dir}")
        return []

    logging.info(f"Loading text files from: {corpus_dir.resolve()}")
    file_count = 0
    total_chunks = 0

    for filepath in corpus_dir.glob("*.txt"):
        file_count += 1
        logging.info(f"Processing file: {filepath.name}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                 logging.warning(f"File '{filepath.name}' is empty or contains only whitespace after cleaning. Skipping.")
                 continue

            chunks = split_text_into_chunks(cleaned_text, chunk_size, chunk_overlap)
            logging.info(f" -> Split into {len(chunks)} chunks.")
            total_chunks += len(chunks)

            for i, chunk_text in enumerate(chunks):
                chunk_data = {
                    "source": filepath.name, # Store the original filename
                    "chunk_id": f"{filepath.stem}_chunk_{i+1}", # Unique ID for the chunk
                    "text": chunk_text
                }
                chunked_corpus.append(chunk_data)

        except Exception as e:
            logging.error(f"Error processing file {filepath.name}: {e}", exc_info=True)

    logging.info(f"Finished processing {file_count} files.")
    logging.info(f"Total chunks created: {total_chunks}")
    return chunked_corpus

# --- Main Execution ---

if __name__ == "__main__":
    logging.info("Starting corpus loading and chunking process...")

    # Create the corpus directory if it doesn't exist (for user convenience)
    if not CORPUS_DIR.exists():
        logging.warning(f"Corpus directory '{CORPUS_DIR}' not found. Creating it.")
        logging.warning(f"Please place your .txt article files inside '{CORPUS_DIR}' before running again.")
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        # Exit if the directory was just created, as it will be empty
        exit()


    # Load and chunk the documents
    chunked_data = load_and_chunk_corpus(CORPUS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

    # Save the chunked data to a JSON file
    if chunked_data:
        logging.info(f"Saving {len(chunked_data)} chunks to {OUTPUT_JSON.resolve()}")
        try:
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(chunked_data, f, ensure_ascii=False, indent=2)
            logging.info("Chunking complete. Output saved.")
        except IOError as e:
            logging.error(f"Failed to save chunked data to {OUTPUT_JSON}: {e}")
    else:
        logging.warning("No chunks were generated. Output file not created.")
