from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

def create_vector_store(documents):
    """
    Create a vector store using OpenAIEmbeddings and Chroma.
    """
    # Ensure OpenAI API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Create and return the vector store
    return Chroma.from_documents(documents, embeddings)

def get_retrieval_qa_chain(vector_store):
    """
    Create a RetrievalQA chain using a vector store and a prompt template.
    """
    # Configure the LLM with parameters
    llm = OpenAI(
        model_name="text-davinci-003",
        temperature=0.7,
        max_tokens=500,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.4,
    )

    # Read the custom prompt from file
    try:
        with open("prompts/summarize_prompt.txt", "r") as prompt_file:
            prompt_string = prompt_file.read()
    except FileNotFoundError:
        raise FileNotFoundError("The file 'prompts/summarize_prompt.txt' was not found.")

    # Create the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=prompt_string,
    )

    # Create the retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Create and return the RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
