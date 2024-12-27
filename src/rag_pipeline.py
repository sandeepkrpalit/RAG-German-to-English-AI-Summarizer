from langchain_openai import OpenAI
from langchain.schema import Document
import logging
from src.pdf_handlers import extract_text_from_pdf
from src.translator import translate_text
from src.embeddings_store import create_vector_store, get_retrieval_qa_chain

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_pdf(pdf_file, user_query="Please summarize the document."):
    """
    Processes a PDF file to extract text, translate it to English, generate embeddings,
    and use a retrieval-based QA chain to answer the user's query.
    """
    try:
        # Step 1: Extract text from the PDF
        logger.info("Extracting text from PDF...")
        documents = extract_text_from_pdf(pdf_file)
        if not documents or all(not doc.page_content for doc in documents):
            logger.warning("No valid text content found in the PDF.")
            return "No valid text content found in the PDF.", []
        logger.info(f"Documents Extracted: {len(documents)} pages")

        # Step 2: Translate text to English
        logger.info("Translating documents to English...")
        translated_docs = [translate_text(doc.page_content) for doc in documents if doc.page_content]
        if not translated_docs:
            logger.warning("No text could be translated.")
            return "No text could be translated.", []
        logger.info(f"Translated Documents: {len(translated_docs)} pages")

        # Step 3: Create Document objects for the vector store
        documents_for_store = [Document(page_content=doc) for doc in translated_docs]
        logger.info("Documents prepared for vector store.")

        # Step 4: Create vector store (Chroma) and retriever
        logger.info("Creating vector store and retriever...")
        vector_store = create_vector_store(documents_for_store)

        # If using Chroma, retrieve the correct retriever method
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        logger.info("Retriever created successfully.")

        # Step 5: Create retrieval QA chain
        logger.info("Creating retrieval QA chain...")
        qa_chain = get_retrieval_qa_chain(retriever)
        logger.info("QA chain created successfully.")

        # Step 6: Retrieve relevant documents
        logger.info(f"Querying the retriever for relevant documents...")
        retrieved_docs = retriever.get_relevant_documents(user_query)
        if not retrieved_docs:
            logger.warning("No relevant documents retrieved.")
            return "No relevant documents found for the query.", []

        # Step 7: Create context from retrieved documents
        context = " ".join(doc.page_content for doc in retrieved_docs)
        logger.debug(f"Context for QA: {context[:500]}...")  # Log first 500 characters of context

        # Step 8: Perform query on the QA chain
        logger.info(f"Querying the chain with user query: {user_query}")
        result = qa_chain({"query": user_query, "context": context})  # Pass both query and context
        logger.info("QA Chain query completed.")

        # Step 9: Return the summarized text and sources
        summary = result.get("result", "No summary generated.")
        sources = [doc.page_content for doc in retrieved_docs]

        return summary, sources

    except Exception as e:
        logger.error(f"Error in process_pdf: {e}")
        return f"An error occurred: {e}", []
