import streamlit as st
from src.rag_pipeline import process_pdf
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Display app title and instructions
st.title("LangSumm: German-to-English Summarizer")
st.write("Upload a PDF in German to generate an English summary.")

# PDF file upload widget
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    st.write("Processing your file...")
    try:
        # Process the PDF file using the pipeline
        summary, sources = process_pdf(pdf_file)
        
        # Display the summarized result
        st.subheader("Summary")
        st.write(summary)
        
        # Display the sources from the document
        st.subheader("Sources")
        for source in sources:
            st.write(source)
    
    except Exception as e:
        # Handle any errors that occur during the process
        st.write(f"Error occurred: {e}")
