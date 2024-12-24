import streamlit as st
from app.rag_pipeline import process_pdf

# Streamlit App
st.title("RAG-based PDF Summarization Tool")
uploaded_file = st.file_uploader("Upload a German PDF", type=["pdf"])

if uploaded_file:
    st.write("Processing your file...")
    summary, sources = process_pdf(uploaded_file)

    st.subheader("Summary:")
    st.write(summary)

    st.subheader("Source Documents:")
    for source in sources:
        st.write(source)
