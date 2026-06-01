import re


STOPWORDS = {

    "what",
    "why",
    "how",
    "is",
    "the",
    "are",
    "does",
    "a",
    "an"
}


def clean_query(query):
    
    print(f"Original query: {query}")

    query = query.lower()

    words = re.findall(r"\b\w+\b", query)

    filtered_words = [

        word

        for word in words

        if word not in STOPWORDS
    ]

    processed_query = " ".join(filtered_words)

    print(f"Processed query: {processed_query}")

    return processed_query