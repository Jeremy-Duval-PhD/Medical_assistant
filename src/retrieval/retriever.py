import chromadb
from sentence_transformers import SentenceTransformer

from src.preprocessing.query_processing import clean_query

from src.utils.config import load_config
config = load_config()


class MedicalRetriever:

    def __init__(
        self,
        chroma_path="../vectorstore/chroma_db",
        collection_name="medical_rag",
        embedding_model_name = (
                f"{config['models']['embedding']}"
                )
    ):

        self.client = chromadb.PersistentClient(
            path=chroma_path
        )

        self.collection = self.client.get_collection(
            collection_name
        )

        self.embedding_model = SentenceTransformer(
            embedding_model_name
        )

    def _get_query_embedding(
        self,
        query
    ):    
        processed_query = clean_query(query)

        query_embedding = self.embedding_model.encode(
            processed_query
        )

        return query_embedding

    def retrieve(
        self,
        query,
        top_k=config['retrieval']['top_k']
    ):

        query_embedding = self._get_query_embedding(query)

        results = self.collection.query(

            query_embeddings=[
                query_embedding.tolist()
            ],

            n_results=top_k
        )

        return results

    def retrieve_with_metadata(
        self,
        query,
        top_k=config['retrieval']['top_k']
    ):

        results = self.retrieve(
            query,
            top_k
        )

        formatted_results = []

        for i in range(len(results["documents"][0])):

            formatted_results.append(

                {

                    "text":
                        results["documents"][0][i],

                    "metadata":
                        results["metadatas"][0][i]
                }
            )

        return formatted_results

    def retrieve_recent(
        self,
        query,
        top_k=config['retrieval']['top_k'],
        min_year=2020
    ):

        query_embedding = self._get_query_embedding(query)

        results = self.collection.query(

            query_embeddings=[
                query_embedding.tolist()
            ],

            n_results=top_k,

            where={
                "year": {
                    "$gte": min_year
                }
            }
        )

        return results

    def retrieve_diverse(
        self,
        query,
        top_k=config['retrieval']['top_k']
    ):

        query_embedding = self._get_query_embedding(query)

        results = self.collection.query(

            query_embeddings=[
                query_embedding.tolist()
            ],

            n_results= max(10, top_k * 3)
        )

        seen_pmids = set()

        selected_documents = []
        selected_metadatas = []
        selected_distances = []
        selected_ids = []

        for doc, metadata, distance, doc_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0]
        ):

            pmid = metadata["pmid"]

            if pmid in seen_pmids:
                continue
            else:
                seen_pmids.add(pmid)

                selected_documents.append(doc)
                selected_metadatas.append(metadata)
                selected_distances.append(distance)
                selected_ids.append(doc_id)

            if len(selected_documents) == top_k:
                break

        return {
            "ids": [selected_ids],
            "documents": [selected_documents],
            "metadatas": [selected_metadatas],
            "distances": [selected_distances]
        }