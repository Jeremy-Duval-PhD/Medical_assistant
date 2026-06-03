from src.llm.generator import generate_answer
from src.llm.prompts import build_prompt

def build_context(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    sections = []

    for doc, metadata in zip(documents, metadatas):

        sections.append(
            f"[PMID:{metadata['pmid']}]\n{doc}"
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

    results = retriever.retrieve(
        query=question
    )
    
    context = build_context(results)
    
    prompt = build_prompt(
        question,
        context,
        strict=strict
    )

    if debug:
        print(f"Prompt:\n{prompt}")
        print("\n")
        start = time.time()
    
    answer = generate_answer(prompt)
    
    if debug:
        elapsed = time.time() - start
        print(f"{elapsed:.1f} seconds")

    sources = get_sources(results)
    print(sources)

    answer = answer + "\n" + sources    
    
    return {
        "results": results,
        "question": question,
        "context": context,
        "answer": answer
    }