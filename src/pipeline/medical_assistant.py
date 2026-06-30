import time

from src.llm.generator import generate_answer
from src.llm.prompts import build_prompt
from src.data.article_store import get_articles

def build_context(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    sections = []

    for doc, metadata in zip(documents, metadatas):

        sections.append(
            f"[PMID:{metadata['pmid']}]\n{doc}"
        )

    return "\n\n".join(sections)


def build_article_context(results):

    articles = get_articles()

    metadatas = results["metadatas"][0]

    sections = []

    for metadata in metadatas:

        pmid = metadata["pmid"]

        article = articles.get(pmid)

        if article is None:
            continue

        sections.append(
            f"[PMID:{pmid}]\n"
            f"{article['text']}"
        )

    return "\n\n".join(sections)


def get_sources(results):

    sources = "Sources used:\n"

    for i, metadata in enumerate(
        results["metadatas"][0],
        start=1
    ):

        sources += f"[{i}] {metadata['title']} "
        sources += f"({metadata['year']}) "
        sources += f"PMID:{metadata['pmid']}\n"

    return sources


def ask_medical_assistant(
    question,
    retriever,
    strict=True,
    debug=False
):
    """
    results = retriever.retrieve(
        query=question
    )
    
    context = build_context(results)
    """
    results = retriever.retrieve_reranked(
        query=question
    )

    context = build_article_context(results)
        
    prompt = build_prompt(
        question,
        context,
        strict=strict
    )

    if debug:
        print(f"Prompt:\n{prompt}\n(Prompt size: {len(prompt)})")
        print("\n")
        start = time.time()
    
    answer = generate_answer(prompt)
    
    if debug:
        elapsed = time.time() - start
        print(f"{elapsed:.1f} seconds")

    sources = get_sources(results)
    print(f'\n{sources}')

    answer = answer + "\n" + sources    

    return {
        "results": results,
        "question": question,
        "context": context,
        "answer": answer
    }