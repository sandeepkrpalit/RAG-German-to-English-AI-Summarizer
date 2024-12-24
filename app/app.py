import streamlit as st
from app.rag_pipeline import process_pdf

st.title("LangSumm: German-to-English Summarizer")
st.write("Upload a PDF in German to generate an English summary.")

pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    st.write("Processing your file...")
    summary, sources = process_pdf(pdf_file)
    
    st.subheader("Summary")
    st.write(summary)
    
    st.subheader("Sources")
    for source in sources:
        st.write(source)
