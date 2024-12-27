from PyPDF2 import PdfReader
from langchain.schema import Document

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file and converts it into LangChain Document objects.

    Args:
        pdf_file: The uploaded PDF file object.

    Returns:
        List[Document]: List of LangChain Document objects with text and metadata.
    """
    reader = PdfReader(pdf_file)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text is None or text.strip() == "":
            text = ""  # Ensure text is not None or empty
        else:
            text = text.strip()

        if text:  # Only add non-empty text to documents
            documents.append(Document(page_content=text, metadata={"page": page_num + 1}))

    # If no text was extracted, raise an error
    if not documents:
        raise ValueError("No text extracted from the PDF file. Please check the file or try another one.")

    return documents
