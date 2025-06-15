import streamlit as st
from rag_agent import load_docs, create_vector_store, get_qa_chain
from translator import translate_to_english
import os

st.set_page_config(page_title="CultureCast AI", layout="wide")
st.title("🎭🗣️ CultureCast AI – Folklore Narrator & Cultural Q&A Agent")

uploaded_file = st.file_uploader("📚 Upload a folklore or story PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing the document..."):
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        docs = load_docs(uploaded_file.name)
        db = create_vector_store(docs)
        qa_chain = get_qa_chain(db)

    st.success("✅ Document ready. Ask about your culture or story content!")

    question = st.text_input("🧠 Ask about the story, proverb, or character")
    if question:
        translated = translate_to_english(question)
        with st.spinner("Thinking..."):
            answer = qa_chain.run(translated)
        st.markdown(f"**Answer:** {answer}")
