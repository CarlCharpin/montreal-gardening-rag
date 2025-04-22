import requests
from bs4 import BeautifulSoup
import time
import random
import json
from urllib.parse import urljoin, urlparse
from pathlib import Path
import hashlib
import logging
import os # Added for potential email env var

# --- Configuration ---
BASE_URL = "https://espacepourlavie.ca"
INDEX_URL = "https://espacepourlavie.ca/en/green-pages"
# Be polite: identify your bot. Consider setting an environment variable for your email.
#CONTACT_EMAIL = os.environ.get('SCRAPER_CONTACT_EMAIL', 'your_email@example.com') # Use env var or default
USER_AGENT = f'MontrealGardeningRAGBot/0.1 (Non-commercial learning project)'
CACHE_DIR = Path("html_cache") # Directory to store downloaded HTML files
OUTPUT_JSON = Path("espacepourlavie_greenpages_corpus.json")
MIN_DELAY = 2.5 # Minimum seconds between requests (increased slightly)
MAX_DELAY = 6.0 # Maximum seconds between requests (increased slightly)
REQUEST_TIMEOUT = 30 # Seconds to wait for server response

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def get_cache_filename(url: str) -> Path:
    """Creates a safe filename for caching based on the URL's hash."""
    # Use SHA256 hash of the URL to create a unique filename
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    # Use the first 16 characters of the hash for brevity
    return CACHE_DIR / f"{url_hash[:16]}.html"

def fetch_and_cache(url: str, session: requests.Session, cache_path: Path) -> bytes | None:
    """
    Fetches a URL if not cached, saves its content to the cache path,
    and returns the content. Includes polite delay. Returns None on failure.
    """
    if cache_path.exists():
        logging.info(f"Cache hit: Loading from {cache_path}")
        try:
            return cache_path.read_bytes()
        except IOError as e:
            logging.error(f"Error reading cache file {cache_path}: {e}")
            # Attempt to re-fetch if cache read fails
            pass # Fall through to fetching logic

    logging.info(f"Cache miss: Fetching {url}")
    # Respectful delay *before* making the request
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    logging.debug(f"Sleeping for {delay:.2f} seconds...")
    time.sleep(delay)

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Save to cache
        logging.info(f"Saving to cache: {cache_path}")
        # Ensure cache directory exists before writing
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
        return response.content

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None
    except IOError as e:
        logging.error(f"Failed to write cache file {cache_path}: {e}")
        # Return content even if caching failed, maybe it's still usable
        return response.content if 'response' in locals() else None


def extract_content(html_content: bytes, url: str) -> dict | None:
    """
    Parses HTML content and extracts title and main text.
    Returns a dictionary or None if extraction fails.
    """
    try:
        # Use lxml parser for speed and robustness if available, fallback to html.parser
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except ImportError:
            logging.warning("lxml not found, falling back to html.parser.")
            soup = BeautifulSoup(html_content, 'html.parser')


        # Extract Title (usually the main H1 tag on the page)
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

        # Extract Main Content (based on inspection, 'div.zone-texte' seems correct)
        # This might need adjustment if the website structure changes.
        content_div = soup.find('div', class_='zone-texte')

        if content_div:
            # Get text, use '\n' as separator for paragraphs, strip whitespace
            # Iterate through relevant tags (p, ul, ol, h2, h3 etc.) if needed for better structure
            text_parts = []
            for element in content_div.find_all(['p', 'h2', 'h3', 'h4', 'ul', 'ol', 'li'], recursive=True):
                 # Avoid extracting text from nested divs or other unwanted elements if any
                 if element.find_parent('div', class_='zone-texte') == content_div:
                     text_parts.append(element.get_text(strip=True))
            
            text = '\n\n'.join(filter(None, text_parts)) # Join parts with double newline, filter empty strings

        else:
            text = "" # Default to empty string if content div not found
            logging.warning(f"Content div 'zone-texte' not found for: {url}")

        # Basic check if extraction was meaningful
        if not text.strip() and title == "No Title Found":
             logging.warning(f"Could not extract meaningful title or text content from: {url}")
             return None # Return None if extraction failed significantly

        return {
            'url': url,
            'title': title,
            'text': text.strip() # Final strip for the whole text block
        }
    except Exception as e:
        logging.error(f"Error parsing HTML from {url}: {e}")
        return None

# --- Main Scraping Logic ---

def main():
    """Main function to orchestrate the scraping process."""
    logging.info("Starting scraper...")
    logging.info(f"Using User-Agent: {USER_AGENT}")

    # Ensure cache directory exists
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Cache directory: {CACHE_DIR.resolve()}")
    except OSError as e:
        logging.error(f"Could not create cache directory {CACHE_DIR}: {e}. Caching disabled.")
        # Allow script to continue without caching if directory creation fails

    # Use a session object to persist headers and potentially cookies
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})

    article_urls = set() # Use a set to automatically handle duplicate URLs

    # 1. Get URLs from the index page
    logging.info(f"--- Step 1: Fetching index page: {INDEX_URL} ---")
    index_cache_path = get_cache_filename(INDEX_URL)
    index_html = fetch_and_cache(INDEX_URL, session, index_cache_path)

    if not index_html:
        logging.error(f"Failed to fetch or load index page {INDEX_URL}. Exiting.")
        return

    try:
        index_soup = BeautifulSoup(index_html, 'lxml' if 'lxml' in BeautifulSoup.features else 'html.parser')
        # Selector based on current structure (April 2025): finds links within the green cards
        # Adjust selector if the website structure changes.
        link_tags = index_soup.select('div.card-tout-vert a')

        for tag in link_tags:
            href = tag.get('href')
            if href:
                # Construct absolute URL if it's relative
                absolute_url = urljoin(BASE_URL, href)
                # Basic validation: ensure it's an HTTP/HTTPS URL and within the same domain
                parsed_url = urlparse(absolute_url)
                if parsed_url.scheme in ['http', 'https'] and parsed_url.netloc == urlparse(BASE_URL).netloc:
                     article_urls.add(absolute_url)
                else:
                    logging.warning(f"Skipping invalid or external URL: {absolute_url}")

        logging.info(f"Found {len(article_urls)} unique article URLs from index page.")

    except Exception as e:
        logging.error(f"Error parsing index page HTML: {e}")
        # Continue if some URLs were found, otherwise exit
        if not article_urls:
            return

    if not article_urls:
        logging.warning("No valid article URLs found on the index page. Exiting.")
        return

    # 2. Process each article URL
    logging.info(f"--- Step 2: Processing {len(article_urls)} articles ---")
    corpus_data = []
    total_urls = len(article_urls)
    # Sort URLs for deterministic processing order (optional but good practice)
    sorted_urls = sorted(list(article_urls))

    for i, article_url in enumerate(sorted_urls):
        logging.info(f"--- Processing article {i+1}/{total_urls}: {article_url} ---")
        cache_file = get_cache_filename(article_url)
        html_content = fetch_and_cache(article_url, session, cache_file)

        if html_content:
            article_data = extract_content(html_content, article_url)
            if article_data:
                corpus_data.append(article_data)
                logging.info(f"Successfully extracted: '{article_data.get('title', 'N/A')}'")
            else:
                 logging.warning(f"Skipping article due to extraction failure: {article_url}")
        else:
            logging.warning(f"Skipping article due to fetch failure: {article_url}")


    # 3. Save the collected data
    logging.info(f"--- Step 3: Saving results ---")
    logging.info(f"Scraping finished. Successfully extracted data from {len(corpus_data)} out of {total_urls} articles.")

    if corpus_data:
        try:
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(corpus_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Corpus saved successfully to {OUTPUT_JSON.resolve()}")
        except IOError as e:
            logging.error(f"Failed to save corpus to {OUTPUT_JSON}: {e}")
    else:
        logging.warning("No data was successfully extracted. Output file not created.")

if __name__ == "__main__":
    # Example of setting contact email via environment variable:
    # export SCRAPER_CONTACT_EMAIL="myrealemail@provider.com"
    # python your_script_name.py
    main()
