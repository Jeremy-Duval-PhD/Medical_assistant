import pandas as pd

_ARTICLES = None


def get_articles():

    global _ARTICLES

    if _ARTICLES is None:

        df = pd.read_json(
            "data/processed/clean_documents.jsonl",
            lines=True
        )

        _ARTICLES = (
            df
            .set_index("pmid")
            .to_dict("index")
        )

    return _ARTICLES