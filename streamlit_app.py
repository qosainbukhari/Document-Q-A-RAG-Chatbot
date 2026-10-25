import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://backend:8000")


def error_message(response):

    # FastAPI's HTTPException puts the message under "detail"
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text



st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

st.title("📄 Document Q&A RAG Chatbot")


# Upload section
st.header("Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file,
            "application/pdf"
        )
    }

    if st.button("Upload PDF"):

        with st.spinner("Uploading and processing PDF..."):

            response = requests.post(
                f"{API_URL}/upload",
                files=files
            )

            if not response.ok:

                st.error(error_message(response))

            else:

                result = response.json()

                st.success(result["message"])

                st.write(
                    f"Chunks created: {result['total_chunks']}"
                )


# Chat section
st.header("Ask Questions")

question = st.text_input(
    "Enter your question"
)

if st.button("Ask"):

    if question.strip():

        with st.spinner("Generating answer..."):

            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "question": question,
                    "session_id": st.session_state.get("session_id")
                }
            )

            if not response.ok:

                st.error(error_message(response))

            else:

                result = response.json()

                # Remember our session so follow-up questions keep their history
                st.session_state["session_id"] = result["session_id"]

                st.subheader("Answer")

                st.write(result["answer"])

                st.subheader("Sources")

                for source in result["sources"]:

                    st.write(source)

    else:

        st.warning("Please enter a question.")