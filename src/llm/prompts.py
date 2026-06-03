

def build_prompt(
    question,
    context,
    strict=True
):

    if strict:

        return f"""
You are a medical assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    return f"""
You are a medical assistant.

Use the provided documents when relevant.

Context:
{context}

Question:
{question}

Answer:
"""