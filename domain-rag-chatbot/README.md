# Domain-Specific RAG Chatbot for PDF Question Answering

##  Project Overview

This project is a Domain-Specific Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF document and ask questions about its content.

The chatbot retrieves relevant information from the uploaded document and uses a Large Language Model (LLM) to generate an answer based only on the retrieved context.

The system also displays the source document and page number so that users can verify the information.

---

##  Objective

The main objective of this project is to build a reliable document question-answering system using Retrieval-Augmented Generation.

The system is designed to:

* Accept PDF documents from users
* Extract text from PDF pages
* Split the extracted text into smaller chunks
* Generate embeddings for the chunks
* Store embeddings in a FAISS vector database
* Retrieve relevant chunks for a user question
* Generate an answer using the retrieved context
* Display the source document and page number
* Avoid generating unsupported information

---

##  RAG Workflow

The system follows this workflow:

**PDF Upload → Text Extraction → Text Chunking → Embeddings → FAISS Vector Store → Question → Relevant Chunk Retrieval → LLM → Grounded Answer + Source**

---

##  Technologies Used

* **Python**
* **Streamlit** – User interface
* **PyPDF** – PDF text extraction
* **LangChain Text Splitters** – Text chunking
* **Sentence Transformers** – Text embeddings
* **all-MiniLM-L6-v2** – Pre-trained embedding model
* **FAISS** – Vector similarity search
* **Google Gemini API** – Answer generation
* **python-dotenv** – Environment variable management
* **Git & GitHub** – Version control

---

##  Project Structure

```text
domain-rag-chatbot/
│
├── documents/
│   └── Uploaded PDF files
│
├── vector_store/
│   └── Vector database files
│
├── app.py
├── document_loader.py
├── vector_store.py
├── rag_pipeline.py
├── prompt.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Main Components

### 1. PDF Processing

The uploaded PDF is processed page by page using PyPDF.

Each page is stored together with metadata such as:

* Page number
* Source document
* Extracted text

### 2. Text Chunking

Large sections of extracted text are divided into smaller chunks using a recursive text splitter.

The current configuration uses:

* Chunk size: 1000 characters
* Chunk overlap: 150 characters

The overlap helps preserve context between neighbouring chunks.

### 3. Embeddings

Each text chunk is converted into a numerical vector using the `all-MiniLM-L6-v2` Sentence Transformer model.

These vectors represent the semantic meaning of the text.

### 4. FAISS Vector Database

The generated embeddings are stored in a FAISS index.

When the user asks a question, the question is also converted into an embedding and compared with the stored vectors to retrieve relevant document chunks.

### 5. Answer Generation

The retrieved chunks are provided to the Gemini model together with the user's question.

The prompt instructs the model to answer only from the supplied document context.

If the required information is unavailable, the chatbot should indicate that it could not find the information in the uploaded documents.

### 6. Source Information

The chatbot displays the page number and source document associated with the retrieved information.

This makes the generated answer easier to verify.

---

##  User Interface

The application is built using Streamlit.

Users can:

1. Upload a PDF
2. Process the PDF
3. Enter a question
4. Receive an answer
5. View the relevant source pages

---

##  Security

API keys are stored in an environment file rather than directly inside the source code.

The `.env` file is excluded from Git using `.gitignore`.

**API keys must never be committed to GitHub.**

---

##  Testing

The system is tested for:

* Retrieval accuracy
* Answer correctness
* Source/page correctness
* Questions with different wording
* Questions whose answers are not present in the document
* Groundedness of generated answers

A testing sheet containing multiple questions will be maintained as part of the project documentation.

---

##  Limitations

* The chatbot depends on the availability and quota of the Gemini API.
* The quality of answers depends on the quality of PDF text extraction and retrieval.
* Scanned PDFs may require OCR for accurate text extraction.
* The system should not be treated as an authoritative source for high-stakes decisions.

---

##  Future Improvements

Possible future enhancements include:

* Support for multiple PDF documents
* OCR for scanned PDFs
* Conversation memory
* Document filtering
* FastAPI backend
* User authentication
* Docker deployment
* Feedback and evaluation system
* Improved retrieval and ranking

---

## ▶ How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd domain-rag-chatbot
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the API key

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

Do not upload the `.env` file to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in the browser.

---

##  Project Status

The core RAG pipeline has been implemented and tested with an Electronics and Computer Engineering PDF.

The application successfully extracts document content, retrieves relevant information, generates grounded answers, and displays source page information.
