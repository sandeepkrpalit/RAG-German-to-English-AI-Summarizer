from app.pdf_handlers import extract_text_from_pdf
from app.translator import translate_text
from app.embeddings_store import create_vector_store, get_retrieval_qa_chain

def process_pdf(pdf_file):
    # Step 1: Extract text from PDF
    documents = extract_text_from_pdf(pdf_file)

    # Step 2: Translate to English
    translated_docs = [
        translate_text(doc.page_content, src="de", dest="en") for doc in documents
    ]

    # Step 3: Create vector store and retrieval chain
    vector_store = create_vector_store(translated_docs)
    qa_chain = get_retrieval_qa_chain(vector_store)

    # Step 4: Summarize using advanced prompting
    query = """
    Step 1: Identify the main idea of the document.
    Step 2: Break down the document into sections and summarize each.
    Step 3: Combine the individual summaries into a cohesive, concise output.
    Final Summary: Summarize this document in English.
    """
    result = qa_chain({"query": query})

    # Extract results
    summary = result["result"]
    sources = [doc.page_content for doc in result["source_documents"]]

    return summary, sources
