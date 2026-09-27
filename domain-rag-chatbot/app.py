import streamlit as st
from dotenv import load_dotenv
import os

from document_loader import extract_text_from_pdf, split_documents
from vector_store import create_vector_store, search_vector_store
from rag_pipeline import generate_answer

load_dotenv()


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 40px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .info-box {
            padding: 18px;
            border-radius: 10px;
            background-color: #f5f7fa;
            border: 1px solid #e1e5ea;
            margin-bottom: 20px;
        }

        .source-box {
    padding: 12px;
    border-radius: 8px;
    background-color: #f8f9fb;
    border-left: 4px solid #4a90e2;
    margin-bottom: 8px;
    color: #222222;
}
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- SESSION STATE ----------------

if "index" not in st.session_state:
    st.session_state.index = None

if "model" not in st.session_state:
    st.session_state.model = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">📚 Domain-Specific RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions from your uploaded PDF using Retrieval-Augmented Generation.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success(f"Selected: {uploaded_file.name}")

        if st.button(
            "⚙️ Process PDF",
            use_container_width=True
        ):

            with st.spinner("Processing document..."):

                pdf_path = os.path.join(
                    "documents",
                    uploaded_file.name
                )

                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                documents = extract_text_from_pdf(
                    pdf_path
                )

                chunks = split_documents(
                    documents
                )

                index, model, chunks = create_vector_store(
                    chunks
                )

                st.session_state.index = index
                st.session_state.model = model
                st.session_state.chunks = chunks
                st.session_state.pdf_name = uploaded_file.name

            st.success(
                "PDF processed successfully! ✅"
            )

    st.divider()

    st.header("🔎 How it works")

    st.write("1. Upload a PDF")
    st.write("2. Extract document text")
    st.write("3. Split text into chunks")
    st.write("4. Create embeddings")
    st.write("5. Retrieve relevant content")
    st.write("6. Generate a grounded answer")

    st.divider()

    st.caption(
        "Powered by Streamlit, FAISS, "
        "Sentence Transformers and Gemini."
    )


# ---------------- DOCUMENT STATUS ----------------

if st.session_state.index is not None:

    st.markdown(
        f"""
        <div class="info-box">
        <b>📖 Active Document:</b> {st.session_state.pdf_name}<br>
        <b>🧩 Indexed Chunks:</b> {len(st.session_state.chunks)}
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.info(
        "👈 Upload a PDF from the sidebar and click "
        "**Process PDF** to begin."
    )


# ---------------- QUESTION AREA ----------------

st.header("💬 Ask a Question")

st.write(
    "Ask a question about the information contained "
    "in the uploaded document."
)

question = st.chat_input(
    "Type your question here..."
)


# ---------------- ANSWER ----------------

if question:

    if st.session_state.index is None:

        st.warning(
            "Please upload and process a PDF before asking a question."
        )

    else:

        # ---------------- USER MESSAGE ----------------

        with st.chat_message("user"):
            st.write(question)


        # ---------------- RETRIEVE RELEVANT CHUNKS ----------------

        retrieved_chunks = search_vector_store(
            question,
            st.session_state.index,
            st.session_state.model,
            st.session_state.chunks,
            top_k=10
        )


        # ---------------- ASSISTANT RESPONSE ----------------

        with st.chat_message("assistant"):

            # Temporary debugging section
            # This helps us check whether RAG retrieval is working.

            st.write("🔎 Retrieved content:")

            for i, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):

                st.write(
                    f"### Result {i} — Page {chunk['page']}"
                )

                st.write(
                    chunk["text"]
                )


            st.divider()


            # ---------------- GENERATE ANSWER ----------------

            with st.spinner(
                "🔍 Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            st.write(answer)


            # ---------------- SOURCES ----------------

            st.subheader("📌 Sources")

            shown_sources = set()

            for chunk in retrieved_chunks:

                source_key = (
                    chunk["source"],
                    chunk["page"]
                )

                if source_key not in shown_sources:

                    st.markdown(
                        f"""
                        <div class="source-box">
                        📄 <b>Page {chunk["page"]}</b><br>
                        {chunk["source"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    shown_sources.add(
                        source_key
                    )