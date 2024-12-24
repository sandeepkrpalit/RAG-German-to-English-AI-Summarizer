from app.pdf_handler import extract_text_from_pdf
from app.translator import translate_text
from app.embeddings_store import create_vector_store, get_retrieval_qa_chain

def process_pdf(pdf_file):
    # Extract text from the PDF
    documents = extract_text_from_pdf(pdf_file)

    # Translate to English
    translated_docs = [
        translate_text(doc.page_content) for doc in documents
    ]

    # Create vector store and retrieval chain
    vector_store = create_vector_store(translated_docs)
    qa_chain = get_retrieval_qa_chain(vector_store)

    # Query with advanced prompting
    with open("prompts/summarize_prompt.txt", "r") as prompt_file:
        query = prompt_file.read()

    result = qa_chain({"query": query})

    # Return the summarized text and sources
    return result["result"], [doc.page_content for doc in result["source_documents"]]
