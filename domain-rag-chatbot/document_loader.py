from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append({
                "text": text,
                "page": page_number,
                "source": pdf_path
            })

    return documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = []

    for document in documents:
        split_texts = text_splitter.split_text(document["text"])

        for text in split_texts:
            chunks.append({
                "text": text,
                "page": document["page"],
                "source": document["source"]
            })

    return chunks