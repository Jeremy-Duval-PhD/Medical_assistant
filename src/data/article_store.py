import pandas as pd
import json

_ARTICLES = None


def get_articles():

    global _ARTICLES
    
    if _ARTICLES is None:
        documents = []
        
        with open("../data/processed/clean_documents.jsonl", "r") as f:
            for line in f:
                documents.append(json.loads(line))

        df = pd.DataFrame(documents)

        _ARTICLES = (
            df
            .set_index("pmid")
            .to_dict("index")
        )

        print(
            f"Loaded {len(_ARTICLES)} articles."
        )

    return _ARTICLES