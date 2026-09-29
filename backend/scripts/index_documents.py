"""
Knowledge Base Builder (one-time ETL job).

Reads the compliance PDFs from backend/data/, splits them into chunks,
embeds each chunk, and uploads the vectors to Azure AI Search.

Run manually whenever the source PDFs change:
    uv run python backend/scripts/index_documents.py

This is NOT part of the live LangGraph pipeline — it's offline prep that
builds the "memory" the Auditor node queries at runtime.
"""

import glob
import logging
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("indexer")

REQUIRED_ENV_VARS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_INDEX_NAME",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def index_docs() -> None:
    """Extract PDFs -> chunk -> embed -> load into Azure AI Search."""
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        logger.error(f"Missing required .env variables: {missing}")
        return

    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    pdf_paths = glob.glob(os.path.join(data_folder, "*.pdf"))
    if not pdf_paths:
        logger.warning(f"No PDFs found in {data_folder}. Add files and re-run.")
        return
    logger.info(
        f"Found {len(pdf_paths)} PDF(s): {[os.path.basename(p) for p in pdf_paths]}"
    )

    chunks = _load_and_chunk(pdf_paths)
    if not chunks:
        logger.warning("No chunks produced from the PDFs. Nothing to index.")
        return

    _upload_to_search(chunks)


def _load_and_chunk(pdf_paths: list[str]) -> list:
    """Extract text from each PDF and split it into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    all_chunks = []

    for path in pdf_paths:
        name = os.path.basename(path)
        try:
            pages = PyPDFLoader(path).load()
            chunks = splitter.split_documents(pages)
            for chunk in chunks:
                chunk.metadata["source"] = (
                    name  # keep track of which PDF this came from
                )
            all_chunks.extend(chunks)
            logger.info(f"  {name} -> {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")

    return all_chunks


def _upload_to_search(chunks: list) -> None:
    """Embed each chunk and upload the vectors to Azure AI Search."""
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        vector_store = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            embedding_function=embeddings.embed_query,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI / AI Search clients: {e}")
        return

    try:
        logger.info(
            f"Uploading {len(chunks)} chunks to index '{os.getenv('AZURE_SEARCH_INDEX_NAME')}'..."
        )
        vector_store.add_documents(documents=chunks)
        logger.info(f"Indexing complete. {len(chunks)} chunks are now searchable.")
    except Exception as e:
        logger.error(f"Failed to upload documents to Azure AI Search: {e}")


if __name__ == "__main__":
    index_docs()
