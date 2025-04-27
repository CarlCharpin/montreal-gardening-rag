import json
from pathlib import Path
import logging
import time
# Import SentenceTransformer and utility functions
from sentence_transformers import SentenceTransformer, util
import torch # semantic_search works efficiently with PyTorch tensors

# --- Configuration ---
EMBEDDINGS_JSON = Path("corpus_with_embeddings.json") # Input file from embedding_generator_v1
# Make sure this is the SAME model used to generate the embeddings
MODEL_NAME = 'all-MiniLM-L6-v2'
# Number of top relevant chunks to retrieve
TOP_N = 5

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables (Load data once) ---
corpus_data = []
corpus_embeddings = None
model = None

def load_data_and_model():
    """Loads the corpus data, embeddings, and the embedding model."""
    global corpus_data, corpus_embeddings, model

    # --- Load Corpus Data with Embeddings ---
    if not EMBEDDINGS_JSON.exists():
        logging.error(f"Embeddings file not found: {EMBEDDINGS_JSON}")
        logging.error("Please run the embedding generation script first.")
        return False

    logging.info(f"Loading corpus data and embeddings from {EMBEDDINGS_JSON}...")
    try:
        with open(EMBEDDINGS_JSON, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        logging.info(f"Loaded {len(corpus_data)} chunks with embeddings.")
        if not corpus_data:
            logging.warning("Input file contains no data. Exiting.")
            return False
        # Extract embeddings and convert to PyTorch tensor
        embeddings_list = [chunk['embedding'] for chunk in corpus_data]
        corpus_embeddings = torch.tensor(embeddings_list, dtype=torch.float32)
        logging.info(f"Corpus embeddings loaded into tensor shape: {corpus_embeddings.shape}")

    except Exception as e:
        logging.error(f"Error loading or processing {EMBEDDINGS_JSON}: {e}", exc_info=True)
        return False

    # --- Load Embedding Model ---
    logging.info(f"Loading Sentence Transformer model: '{MODEL_NAME}'...")
    try:
        # Load the model onto CPU or GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device}")
        model = SentenceTransformer(MODEL_NAME, device=device)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model '{MODEL_NAME}': {e}")
        logging.error("Ensure 'sentence-transformers' and 'torch' are installed.")
        return False

    return True


def find_relevant_chunks(query: str, top_n: int = TOP_N) -> list:
    """
    Finds the most relevant chunks for a given query using semantic search.

    Args:
        query: The user's question or search term.
        top_n: The number of top chunks to return.

    Returns:
        A list of dictionaries, where each dictionary contains the
        retrieved chunk data ('source', 'chunk_id', 'text') and its
        similarity 'score'. Returns an empty list if data/model not loaded
        or an error occurs.
    """
    global corpus_data, corpus_embeddings, model

    if not corpus_data or corpus_embeddings is None or model is None:
        logging.error("Corpus data, embeddings, or model not loaded. Cannot perform search.")
        return []

    logging.info(f"Searching for top {top_n} chunks relevant to query: '{query}'")
    start_time = time.time()

    try:
        # 1. Embed the query
        query_embedding = model.encode(query, convert_to_tensor=True)

        # 2. Perform semantic search
        # Ensure corpus_embeddings is on the same device as the model/query_embedding
        corpus_embeddings = corpus_embeddings.to(model.device)
        query_embedding = query_embedding.to(model.device)

        hits = util.semantic_search(
            query_embedding,
            corpus_embeddings,
            top_k=top_n
            )

        # semantic_search returns a list of lists (one per query). We only have one query.
        hits = hits[0]
        logging.info(f"Found {len(hits)} potential hits.")

        # 3. Format results
        results = []
        for hit in hits:
            corpus_id = hit['corpus_id'] # Index of the chunk in the original list
            score = hit['score']         # Similarity score
            if 0 <= corpus_id < len(corpus_data):
                retrieved_chunk = corpus_data[corpus_id]
                results.append({
                    "score": score,
                    "source": retrieved_chunk.get("source", "N/A"),
                    "chunk_id": retrieved_chunk.get("chunk_id", "N/A"),
                    "text": retrieved_chunk.get("text", "")
                })
            else:
                logging.warning(f"Invalid corpus_id {corpus_id} found in search results.")

        end_time = time.time()
        logging.info(f"Search took {end_time - start_time:.4f} seconds.")
        return results

    except Exception as e:
        logging.error(f"Error during semantic search: {e}", exc_info=True)
        return []

# --- Main Execution Example ---

if __name__ == "__main__":
    logging.info("Starting retrieval process...")

    # Load data and model once at the start
    if not load_data_and_model():
        exit() # Exit if loading failed

    # --- Example Queries ---
    example_queries = [
        "How do I protect roses over winter?",
        "When should I start tomato seeds indoors in Montreal?",
        "How should I plant my peas?" # Specific test
    ]

    for q in example_queries:
        print("-" * 40)
        retrieved_results = find_relevant_chunks(q, top_n=TOP_N)

        if retrieved_results:
            print(f"\nTop {len(retrieved_results)} results for query: '{q}'\n")
            for i, result in enumerate(retrieved_results):
                print(f"Rank {i+1}: Score={result['score']:.4f}, Source='{result['source']}', Chunk ID='{result['chunk_id']}'")
                print(f"Text: {result['text'][:250]}...\n") # Print start of the chunk text
        else:
            print(f"\nNo results found for query: '{q}'")
        print("-" * 40)

