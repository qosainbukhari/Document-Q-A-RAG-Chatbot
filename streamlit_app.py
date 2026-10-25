import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")


st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide"
)


# ---------- talking to the FastAPI backend ----------

def api(method, path, timeout=120, **kwargs):

    # Returns (data, error_message); exactly one of them is None
    try:
        response = requests.request(
            method, f"{API_URL}{path}", timeout=timeout, **kwargs
        )

    except requests.ConnectionError:
        return None, f"Cannot reach the backend at {API_URL}. Is it running?"

    except requests.Timeout:
        return None, "The backend took too long to answer. Please try again."

    if not response.ok:

        # FastAPI's HTTPException puts the message under "detail"
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        # Validation errors (422) come as a list of problems
        if isinstance(detail, list):
            detail = "; ".join(item.get("msg", str(item)) for item in detail)

        return None, f"{detail} (HTTP {response.status_code})"

    return response.json(), None


# ---------- chat state (kept per browser tab) ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None


def page_order(page):

    # Numbers in numeric order (3 before 22); anything else ("N/A") last
    return (0, page) if isinstance(page, int) else (1, 0)


def format_sources(sources):

    # [{source: a.pdf, page: 3}, {source: a.pdf, page: 5}] -> "📄 a.pdf — p. 3, 5"
    pages = {}

    for item in sources:
        pages.setdefault(item["source"], []).append(item["page"])

    return "  ·  ".join(
        f"📄 **{source}** — p. {', '.join(str(p) for p in sorted(page_list, key=page_order))}"
        for source, page_list in pages.items()
    )


def show_message(message):

    with st.chat_message(message["role"]):

        if message.get("error"):
            st.error(message["content"])
            return

        st.markdown(message["content"])

        # Show the rewritten question when a follow-up was made standalone
        standalone = message.get("standalone_question")

        if standalone and standalone != message.get("question"):
            st.caption(f"🔎 Searched for: *{standalone}*")

        if message.get("sources"):
            st.caption(format_sources(message["sources"]))


# ---------- sidebar: backend status, upload, documents ----------

with st.sidebar:

    st.header("📚 Documents")

    _, health_error = api("GET", "/health", timeout=3)

    if health_error:
        st.error(health_error)
    else:
        st.success("Backend online", icon="✅")

    uploaded_file = st.file_uploader("Add a PDF", type=["pdf"])

    if st.button(
        "Upload and index",
        type="primary",
        disabled=uploaded_file is None or bool(health_error),
        use_container_width=True
    ):

        with st.status(f"Indexing {uploaded_file.name}...", expanded=False) as status:

            result, error = api(
                "POST", "/upload",
                timeout=600,  # big PDFs (or multi-vector summaries) take a while
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )

            if error:
                status.update(label="Upload failed", state="error")
                st.error(error)
            else:
                status.update(label=f"Indexed {result['filename']}", state="complete")
                st.toast(
                    f"{result['filename']}: {result['pages']} pages, "
                    f"{result['total_chunks']} chunks"
                )

    if not health_error:

        documents, error = api("GET", "/documents", timeout=10)

        if error:
            st.warning(error)

        elif documents:
            st.subheader("Indexed")
            for document in documents:
                st.markdown(
                    f"**{document['source']}**  \n"
                    f"<small>{document['pages']} pages · {document['chunks']} chunks</small>",
                    unsafe_allow_html=True
                )

        else:
            st.info("No documents yet. Upload a PDF to start.")

    st.divider()

    if st.button("🧹 New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id[:8]}`")


# ---------- main area: the conversation ----------

st.title("📄 Document Q&A")
st.caption(
    "Answers come only from your uploaded PDFs, with page citations. "
    "Follow-up questions remember the conversation."
)

if not st.session_state.messages:
    st.info(
        "Upload a PDF in the sidebar, then ask something like:\n\n"
        "- *Summarise the main ideas of this document.*\n"
        "- *What does it say about ...?*\n"
        "- then a follow-up: *Can you explain that more simply?*"
    )

for message in st.session_state.messages:
    show_message(message)

question = st.chat_input("Ask a question about your documents")

if question:

    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    with st.spinner("Searching your documents..."):

        result, error = api(
            "POST", "/chat",
            json={"question": question, "session_id": st.session_state.session_id}
        )

    if error:
        reply = {"role": "assistant", "content": error, "error": True}

    else:
        # Remember our session so follow-up questions keep their history
        st.session_state.session_id = result["session_id"]

        reply = {
            "role": "assistant",
            "content": result["answer"],
            "question": question,
            "standalone_question": result.get("standalone_question"),
            "sources": result.get("sources", []),
        }

    st.session_state.messages.append(reply)
    show_message(reply)
