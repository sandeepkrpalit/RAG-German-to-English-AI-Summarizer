from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(documents, embeddings)

def get_retrieval_qa_chain(vector_store):
    # Configure the LLM with advanced parameters
    llm = OpenAI(
        model_name="text-davinci-003",
        temperature=0.7,        # Balanced creativity
        max_tokens=500,         # Restrict output length
        top_p=0.9,              # Nucleus sampling
        frequency_penalty=0.2,  # Reduce repetition
        presence_penalty=0.4    # Encourage new ideas
    )
    
    # Read the custom prompt
    with open("prompts/summarize_prompt.txt", "r") as prompt_file:
        prompt_template = prompt_file.read()

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
