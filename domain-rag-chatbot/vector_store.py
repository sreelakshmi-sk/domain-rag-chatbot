from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def create_vector_store(chunks):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, model, chunks


def search_vector_store(
    question,
    index,
    model,
    chunks,
    top_k=5
):

    question_lower = question.lower()

    # -------------------------------------------------
    # SPECIAL HANDLING FOR MODULE 1 QUESTIONS
    # -------------------------------------------------

    if (
        "module 1" in question_lower
        and "electronic devices" in question_lower
        and "circuits" in question_lower
    ):

        matching_chunks = []

        for chunk in chunks:

            text = chunk["text"].lower()

            score = 0

            # -------------------------------------------------
            # MODULE 1 INDICATORS
            # -------------------------------------------------

            if "module no. 1" in text:
                score += 8

            if "module 1" in text:
                score += 8

            if "module i" in text:
                score += 8

            # -------------------------------------------------
            # SUBJECT INDICATORS
            # -------------------------------------------------

            if "electronic devices and circuits" in text:
                score += 10

            elif "electronic devices" in text:
                score += 6

            elif "electronic device" in text:
                score += 4

            # -------------------------------------------------
            # SYLLABUS INDICATORS
            # -------------------------------------------------

            if "syllabus" in text:
                score += 3

            if "contact hours" in text:
                score += 2

            # -------------------------------------------------
            # ONLY KEEP STRONGLY RELEVANT CHUNKS
            # -------------------------------------------------

            if score >= 8:

                matching_chunks.append(
                    {
                        "chunk": chunk,
                        "score": score
                    }
                )

        # -------------------------------------------------
        # SORT BY RELEVANCE
        # -------------------------------------------------

        matching_chunks.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # -------------------------------------------------
        # RETURN BEST MATCHES
        # -------------------------------------------------

        if matching_chunks:

            return [
                item["chunk"]
                for item in matching_chunks[:top_k]
            ]

    # -------------------------------------------------
    # NORMAL SEMANTIC SEARCH
    # -------------------------------------------------

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    # Search more candidates first
    candidate_k = min(
        50,
        len(chunks)
    )

    distances, indices = index.search(
        question_embedding,
        candidate_k
    )

    results = []

    for position, i in enumerate(indices[0]):

        chunk = chunks[i]

        results.append(
            {
                "chunk": chunk,
                "distance": distances[0][position]
            }
        )

    # Smaller L2 distance = more similar
    results.sort(
        key=lambda x: x["distance"]
    )

    return [
        result["chunk"]
        for result in results[:top_k]
    ]