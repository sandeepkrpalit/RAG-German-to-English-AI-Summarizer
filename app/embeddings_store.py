from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def create_vector_store(documents):
    """
    Creates a vector store using OpenAI embeddings.

    Args:
        documents: List of translated documents.

    Returns:
        Chroma: Vector store for retrieval.
    """
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(documents, embeddings)

def get_retrieval_qa_chain(vector_store):
    """
    Builds a retrieval-based QA chain.

    Args:
        vector_store: Vector store for document retrieval.

    Returns:
        RetrievalQA: QA chain for querying the vector store.
    """
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI

    llm = OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=500)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
