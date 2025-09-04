import streamlit as st
import os
from pdf_loader import load_and_split_pdf
from vectorstore import create_or_load_vectorstore, pdf_hash
from llm_chain import create_conv_chain

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with PDF (Persistent + Memory + Source Pages)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF and generating embeddings…"):
        # Generate unique ID for PDF
        pdf_id = pdf_hash(uploaded_file)

        # Split PDF into chunks
        texts = load_and_split_pdf(uploaded_file)

        # Load or create vectorstore
        vectorstore_instance = create_or_load_vectorstore(texts, pdf_id)

        # Create conversation chain
        conv_chain = create_conv_chain(vectorstore_instance)

    st.subheader("📑 PDF Loaded")
    st.text(f"Number of chunks: {len(texts)}")

    # Generate summary
    st.subheader("📝 PDF Summary")
    summary_prompt = "Please provide a concise summary of the PDF content."
    summary_result = conv_chain({"question": summary_prompt})
    st.write(summary_result['answer'])

    # Interactive Q&A
    st.subheader("💬 Ask questions about the PDF")
    user_input = st.text_input("Type your question here:")

    if user_input:
        chat_result = conv_chain({"question": user_input})
        st.write("**Answer:**", chat_result['answer'])

        st.write("**Source Pages:**")
        for doc in chat_result['source_documents']:
            st.write(doc.metadata.get("source"))
