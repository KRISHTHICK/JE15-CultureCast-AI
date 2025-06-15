from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

def load_docs(path):
    loader = PyPDFLoader(path)
    return loader.load()

def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    return db

def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    llm = Ollama(model="tinyllama")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
