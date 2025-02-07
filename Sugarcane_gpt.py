import os
import requests
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Constants
FAISS_DB_DIR = 'faiss_index_sugarcane'
PROCESSED_LINKS_FILE = 'processed_links_sugarcane.txt'
MAX_RETRIES = 3
RETRY_DELAY = 5

# GPT-4o Client
gpt_client = AzureChatOpenAI(
    openai_api_key="51hXldrb7aZiGtgsfyCGSXj7hzfdzSRAbeNitNhu1CvNWd1dbmodJQQJ99BBACYeBjFXJ3w3AAABACOGPXTz", 
    azure_endpoint="https://linkchatbot-api.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview"
)

# Embeddings Client 
embeddings_client = AzureOpenAIEmbeddings(
    openai_api_key="51hXldrb7aZiGtgsfyCGSXj7hzfdzSRAbeNitNhu1CvNWd1dbmodJQQJ99BBACYeBjFXJ3w3AAABACOGPXTz",
     azure_endpoint="https://linkchatbot-api.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15",
    model="text-embedding-3-large",
    openai_api_version="2023-05-15"
)
embeddings_cache ={}

def create_embedding(text):
    try:
        embedding = embeddings_client.embed_query(text)  # Create embedding
        embeddings_cache[text] = embedding  # Store in cache (if using a cache)
        return embedding
    except Exception as e:
        print(f"Embedding error for '{text}': {str(e)}")
        return None
    
# Load Processed Links
def get_processed_links():
    return set(open(PROCESSED_LINKS_FILE).read().splitlines()) if os.path.exists(PROCESSED_LINKS_FILE) else set()

def mark_link_as_processed(url):
    with open(PROCESSED_LINKS_FILE, 'a') as file:
        file.write(url + '\n')

# Web Scraping Function with Error Handling
def scrape_website(url):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove ads
            for ad in soup.find_all(['div', 'span'], class_=lambda x: x and 'ad' in x.lower()):
                ad.decompose()

            # Extract text content
            text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])

            # Extract tables
            for table in soup.find_all('table'):
                table_text = ' '.join([td.get_text() for td in table.find_all('td')])
                text += f" {table_text}"

            return text
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(RETRY_DELAY)

    print(f"Failed to retrieve {url} after {MAX_RETRIES} attempts.")
    return ""

# Extract Absolute Links from a Page
def get_absolute_links(url):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
            return list(set(filter(lambda link: not link.endswith(('.jpg', '.jpeg', '.png', '.gif')), links)))
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed to retrieve links from {url}: {e}")
            time.sleep(RETRY_DELAY)
    return []

# Create or Load FAISS Vector DB
def initialize_qa_chain(unprocessed_links):
    if os.path.exists(FAISS_DB_DIR):
        load_existing = input("Load existing FAISS vectorDB? (y/n): ").strip().lower()
        if load_existing == 'y':
            print("Loading existing FAISS vectorDB...")

            vectordb = FAISS.load_local(
                FAISS_DB_DIR,
                embeddings_client,
                allow_dangerous_deserialization=True
            )
            print("FAISS vectorDB loaded successfully.")

            return RetrievalQA.from_chain_type(
                llm=gpt_client,
                chain_type="stuff",
                # retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
                retriever=vectordb.as_retriever(search_kwargs={"k": 5, "fetch_k": 10}),
                return_source_documents=True
            )
        
    print("Creating new embeddings from provided URLs...")
    documents = []
    processed_links = get_processed_links()
    vectordb = None

    while unprocessed_links:
        url = unprocessed_links.pop(0)
        if url in processed_links:
            print(f"Skipping already processed URL: {url}")
            continue

        print(f"Processing: {url}")
        website_text = scrape_website(url)
    if website_text:
            print(f"Scraped {len(website_text)} characters from {url}")
            documents.append(Document(page_content=website_text, metadata={"source": url}))
            mark_link_as_processed(url)

    for link in get_absolute_links(url):
                if link not in processed_links:
                    internal_text = scrape_website(link)
                    if internal_text:
                        print(f"Scraped internal content from {link} (length: {len(internal_text)} characters)")
                        documents.append(Document(page_content=internal_text, metadata={"source": link}))
                        mark_link_as_processed(link)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=90)
    splits = text_splitter.split_documents(documents)

    # Create FAISS Vector DB
    if splits:
                vectordb = FAISS.from_documents(splits, embeddings_client)
                vectordb.save_local(FAISS_DB_DIR)
                print(f"FAISS vectorDB saved at {FAISS_DB_DIR}.")
    else:
                print(f"No content found to process for {url}")

    if not documents:
        raise ValueError("No documents were processed, unable to create vector database.")

    return RetrievalQA.from_chain_type(
        llm=gpt_client,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

# Process User Queries
def process_answer(instruction, qa_chain):
    result = qa_chain.invoke({"query": instruction})
    source_docs = result.get('source_documents', [])
    answer = result['result'].strip()

    if len(source_docs) == 0 or any(phrase in answer.lower() for phrase in ["does not provide", "couldn't find", "i don't know", "not mentioned"]):
        return "Sorry, I couldn't find the answer to your question."

    sources = list(set(doc.metadata.get("source", "Unknown source") for doc in source_docs))
    source_links = "\n".join(sources) if sources else "Unknown source"

    return f"{answer.capitalize()}\n\nSources:\n{source_links}"


# Read URLs from File
def get_urls_from_file(filename):
    if not os.path.exists(filename):
        print(f"{filename} not found.")
        return []
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip()]

# Main Function
def main():
    filename = 'sugarcane_links.txt'
    urls = get_urls_from_file(filename)

    if not urls:
        print(f"No URLs found in {filename}. Exiting.")
        return

    print("Processing embeddings. This may take some time...")
    qa_chain = initialize_qa_chain(urls)
    print("Embeddings processed. You can now ask questions about Sugarcane.")

    while True:
        prompt = input("\nYou: ")
        if prompt.lower() in ["exit", "quit", "bye"]:
            print("Exiting the chatbot.")
            break
        response = process_answer(prompt, qa_chain)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
