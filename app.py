import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

# ------------------- Title -------------------
st.title("Zero-Hallucination Policy QA & Citizen Feedback Agent")

# ------------------- PDF Upload -------------------
pdf = st.file_uploader("Upload a Government Policy PDF", type=["pdf"])
if pdf:
    pdf_reader = PdfReader(pdf)
    text = "".join([page.extract_text() for page in pdf_reader.pages])

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small", model_kwargs={"temperature": 0, "max_length": 256}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PromptTemplate(
            input_variables=["context","question"],
            template="Use only the context to answer.\nContext: {context}\nQuestion: {question}\nAnswer:"
        )}
    )

    st.success("✅ Policy Document Processed")

    # ------------------- QA Section -------------------
    query = st.text_input("Ask a policy-related question:")
    if query:
        agent1 = qa_chain.run(query)  # Agent 1: Retrieval Answer
        agent2 = "EVIDENCE OK" if query.lower().split()[0] in agent1.lower() else "EVIDENCE CHECK"  # Agent 2
        agent3 = agent1 if agent2 == "EVIDENCE OK" else "Not enough evidence in policy."
        agent4 = f"Final Answer: {agent3}"  # Agent 3 summarizer
        st.markdown(f"**Agent 1 (Retriever):** {agent1}")
        st.markdown(f"**Agent 2 (Evidence Check):** {agent2}")
        st.markdown(f"**Agent 3 (Verifier):** {agent3}")
        st.markdown(f"**Agent 4 (Final Reply):** {agent4}")

# ------------------- Citizen Feedback -------------------
st.subheader("Citizen Feedback Analysis")
feedbacks = st.text_area("Paste citizen feedback (one per line):")
if feedbacks:
    sia = SentimentIntensityAnalyzer()
    rows, total = [], 0
    for fb in feedbacks.splitlines():
        score = sia.polarity_scores(fb)["compound"]
        rows.append((fb, score))
        total += score
    avg = total / len(rows)

    st.table(rows)
    st.write(f"📊 Avg Sentiment Score: {avg:.2f}")
