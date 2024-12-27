import streamlit as st
from streamlit import session_state
import PyPDF2
from deep_translator import GoogleTranslator
from transformers import pipeline

st.set_page_config(page_title="German PDF to English Summary", layout="wide")

# Load the Hugging Face summarization model
summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")

def translate_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

    try:
        translator = GoogleTranslator(source='de', target='en')
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return None

def summarize_text(text):
    if text:
        with st.spinner("Summarizing..."):
            try:
                summary = summarizer(text, max_length=150, min_length=30)
                return summary[0]['summary_text']
            except Exception as e:
                st.error(f"Summarization failed: {e}")
                return f"An error occurred during summarization: {e}"
    else:
        return "Translation failed. Please try again."

def main():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f0f0; /* Light gray background */
        }
        .st-h1 {
            text-align: center;
            color: #000000; /* Black text color */
            background-color: #D4AF37; /* German flag yellow */
            padding: 10px;
            border-radius: 5px;
        }
        .st-container {
            padding: 20px; 
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align: center;'>German PDF to English Summary</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("**Upload German PDF:**")
        uploaded_file = st.file_uploader("", type="pdf")
        if uploaded_file:
            session_state.uploaded_file = uploaded_file

            if "pdf_text" not in session_state:
                session_state.pdf_text = ""

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Extracting PDF Text..."):
                session_state.pdf_text = ""
                with open("temp.pdf", 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        session_state.pdf_text += f"**Page {page_num + 1}:**\n{page.extract_text()}\n\n" 

    with col2:
        if session_state.pdf_text:
            st.header("German PDF Text")
            st.markdown(f"""
                <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; overflow-y: scroll; max-height: 400px;">
                    {session_state.pdf_text} 
                </div>
            """, unsafe_allow_html=True)

    if session_state.uploaded_file:
        if st.button("Translate & Summarize"):
            with st.spinner("Processing..."):
                session_state.english_text = translate_pdf("temp.pdf")
                session_state.summary = summarize_text(session_state.english_text)

    if session_state.summary:
        st.header("English Summary")
        st.text(session_state.summary)

if __name__ == "__main__":
    main()