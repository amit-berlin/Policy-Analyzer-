import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import numpy as np
import pandas as pd

# ------------------------
# Helper: Extract PDF text
# ------------------------
def extract_pdf_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ------------------------
# Helper: Chunk text
# ------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ------------------------
# Agent 1-3: RAG-lite Chat
# ------------------------
def rag_chat(query, chunks):
    vectorizer = TfidfVectorizer().fit(chunks + [query])
    vectors = vectorizer.transform(chunks + [query])
    sims = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    best_idx = sims.argmax()
    best_chunk = chunks[best_idx]
    confidence = sims[best_idx]
    return best_chunk, confidence

# ------------------------
# Citizen Feedback Analyzer
# ------------------------
def analyze_feedback(feedback_list):
    results = []
    for fb in feedback_list:
        polarity = TextBlob(fb).sentiment.polarity
        sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        results.append({"feedback": fb, "sentiment": sentiment})
    return pd.DataFrame(results)

# ------------------------
# Impact Scoring (Toy ML)
# ------------------------
def compute_impact(before, after):
    try:
        impact = ((after - before) / before) * 100
        return round(impact, 2)
    except ZeroDivisionError:
        return 0

# ------------------------
# Streamlit UI
# ------------------------
st.title("🇮🇳 National AI Policy Intelligence MVP")

st.sidebar.header("Upload Policy PDF")
pdf_file = st.sidebar.file_uploader("Upload a Policy Document", type="pdf")

if pdf_file:
    text = extract_pdf_text(pdf_file)
    chunks = chunk_text(text)

    st.subheader("Ask a Question (Policy RAG)")
    query = st.text_input("Enter your policy-related question:")

    if query:
        answer, score = rag_chat(query, chunks)
        st.markdown("### Agentic Answer")
        st.write(answer)
        st.caption(f"Confidence Score: {round(score,2)}")

    # Citizen Feedback Section
    st.subheader("Citizen Feedback Analysis")
    feedback_input = st.text_area("Paste citizen feedback (one per line):")
    if feedback_input:
        feedback_list = feedback_input.strip().split("\n")
        df = analyze_feedback(feedback_list)
        st.dataframe(df)

    # Impact Scoring
    st.subheader("Impact Measurement")
    before = st.number_input("Metric Before Policy", value=100)
    after = st.number_input("Metric After Policy", value=120)
    if st.button("Compute Impact"):
        impact = compute_impact(before, after)
        st.success(f"Estimated Impact: {impact}% change")

else:
    st.info("⬅️ Upload a policy PDF to start")
