from PyPDF2 import PdfReader
from langchain.schema import Document

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file and converts it into a list of LangChain Document objects.

    Args:
        pdf_file: The uploaded PDF file object.

    Returns:
        List of LangChain Document objects containing text and metadata.
    """
    reader = PdfReader(pdf_file)
    documents = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Create a Document object for each page
            documents.append(Document(page_content=text, metadata={"page": page_num + 1}))

    return documents
