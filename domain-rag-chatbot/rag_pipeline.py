from dotenv import load_dotenv
import os
from google import genai
from google.genai import errors, types

load_dotenv()


def generate_answer(question, retrieved_chunks):
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=1
            )
        )
    )

    context = "\n\n".join(
        [
            f"Source: {chunk['source']}, Page: {chunk['page']}\n"
            f"{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are a document question-answering assistant.

Answer ONLY using the supplied context.

If the answer is not available in the context, say:
"I could not find this information in the uploaded documents."

Do not invent facts.

Mention the source document and page number when available.

Context:
{context}

Question:
{question}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text

    except errors.APIError as e:

        if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
            return (
                "⚠️ Gemini API quota has been reached. "
                "Please try again later."
            )

        if getattr(e, "code", None) == 503 or "UNAVAILABLE" in str(e):
            return (
                "⚠️ Gemini is temporarily unavailable. "
                "Please try again in a moment."
            )

        raise