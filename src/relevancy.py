# import
import os
import datetime
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain.embeddings import FakeEmbeddings
from langchain.embeddings import OpenAIEmbeddings

# create the document and split it into chunks
def create_documents(papers: list):
    docs = []
    for paper in papers:
        summary = paper["title"] + "\n" + paper["abstract"]
        docs.append(
            Document(
                page_content=summary, 
                metadata={
                    "main_page": paper["main_page"],
                    "pdf": paper["pdf"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "subjects": paper["subjects"],
                    "abstract": paper["abstract"],
                    "summary": summary,
                }
            )
        )
    return docs


def split_docs(documents, chunk_size=1000, chunk_overlap=0):    
    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


def embed_documents(docs, type="huggingface", id=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")):
    if type == "huggingface":
        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    elif type == "openai":
        embedding_function = OpenAIEmbeddings()
    else:
        embedding_function = FakeEmbeddings(size=1352)
    
    db = FAISS.from_documents(docs, embedding_function)
    if not os.path.exists("./data"):
        os.makedirs("./data")
    db.save_local(f"./data/faiss_index_{id}")
    return db


def get_relevance_docs_with_score(query, db):
    # query it
    docs = db.similarity_search_with_score(query)
    return docs


def top_relevance_docs(docs, n=10):
    scores = [doc[-1] for doc in docs]
    ranks = sorted(scores, reverse=True)[:n]
    
    top_n = []
    for (doc, score) in docs:
        for rank in ranks:
            if rank == score:
                top_n.append(doc)
    return top_n


def get_top_relevance_paper(papers, query):
    documents = create_documents(papers)
    # docs = split_docs(documents, chunk_size=1000, chunk_overlap=0)
    db = embed_documents(documents)
    relevance_docs = get_relevance_docs_with_score(query, db)
    top_relevance_docs = top_relevance_docs(relevance_docs, n=10)
    return [doc.metadata for doc in top_relevance_docs]
