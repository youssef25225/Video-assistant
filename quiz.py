import streamlit as st
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


@st.cache_resource
def load_qa_pipeline():
    return pipeline(
        "question-answering",
        model="deepset/minilm-uncased-squad2",
        device=-1
    )


def load_qa(transcript_text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    docs = text_splitter.split_text(transcript_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    return retriever


def get_answer(question: str, retriever) -> str:
    qa = load_qa_pipeline()

    related_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in related_docs])[:1200]

    result = qa(
        question=question,
        context=context,
        truncation=True,
        max_seq_len=512
    )

    if result.get("score", 0) < 0.1:
        return "I couldn't find a confident answer in the transcript."

    return result["answer"]