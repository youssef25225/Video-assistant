import streamlit as st
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

CHUNK_SIZE = 250


@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="Falconsai/text_summarization",
        device=-1
    )


def chunk_text(text: str) -> list:
    sentences = sent_tokenize(text)
    chunks = []
    chunk = []
    words = 0

    for sentence in sentences:
        w = len(sentence.split())
        if words + w > CHUNK_SIZE and chunk:
            chunks.append(" ".join(chunk))
            chunk = []
            words = 0
        chunk.append(sentence)
        words += w

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


def summarize(text: str) -> dict:
    if len(text.strip()) < 50:
        return {"points": [], "detailed": "Text too short to summarize."}

    model = load_model()
    chunks = chunk_text(text)

    summaries = []
    for chunk in chunks:
        result = model(
            chunk,
            max_length=120,
            min_length=30,
            truncation=True,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    combined = " ".join(summaries)
    final_chunks = chunk_text(combined)

    final_summaries = []
    for chunk in final_chunks:
        result = model(chunk, max_length=200, min_length=80, truncation=True)
        final_summaries.append(result[0]["summary_text"])

    final = " ".join(final_summaries)
    points = sent_tokenize(final)
    points = sorted(points,key=len, reverse=True)[:5]

    return {
        "points": points,
        "detailed": final
    }