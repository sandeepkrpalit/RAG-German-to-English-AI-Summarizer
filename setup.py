from setuptools import setup, find_packages

setup(
    name="rag_summarization_tool",
    version="1.0.0",
    description="A RAG-based PDF summarization tool for translating and summarizing German PDFs.",
    author="Sandeep Kumar Palit",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "openai",
        "chromadb",
        "sentence-transformers",
        "pypdf2",
        "streamlit",
        "googletrans==4.0.0-rc1"
    ],
)
