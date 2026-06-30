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


    def _get_search_k(
        self, 
        top_k=config["retrieval"]["top_k"]
    ):
        
        search_k = max(
            config["retrieval"]["search_min"],
            top_k * config["retrieval"]["search_multiplier"]
        )

        return search_k


    def _retrieve_candidates(
        self,
        query,
        search_k
    ):
        query_embedding = self._get_query_embedding(query)

        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=search_k
        )


    def _normalize_distances(
        self,
        distances
    ):
        """
        Convert Chroma distances into similarity scores in [0, 1].

        Parameters
        ----------
        distances : list[float]
            Distances returned by ChromaDB.

        Returns
        -------
        list[float]
            Normalized semantic similarity scores.
        """

        min_distance = min(distances)
        max_distance = max(distances)

        if max_distance == min_distance:
            return [1.0] * len(distances)

        return [
            (max_distance - d) / (max_distance - min_distance)
            for d in distances
        ]


    def _compute_score(
        self,
        semantic_score,
        metadata
    ):
        """
        Compute the final retrieval score.

        Parameters
        ----------
        semantic_score : float
            Normalized semantic similarity.

        metadata : dict
            Metadata associated with the retrieved chunk.

        Returns
        -------
        float
            Final reranking score.
        """

        evidence_bonus = (
            metadata["evidence_level"]
            * config["retrieval"]["reranking"]["evidence_weight"]
        )

        return semantic_score + evidence_bonus


    def _unique_articles(
        self,
        results
    ):
        """
        Keep only the best chunk (smallest distance)
        for each PMID.
        """

        articles = {}

        for doc_id, doc, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            pmid = metadata["pmid"]

            if (
                pmid not in articles
                or distance < articles[pmid]["distance"]
            ):

                articles[pmid] = {
                    "id": doc_id,
                    "document": doc,
                    "metadata": metadata,
                    "distance": distance,
                }

        selected = sorted(
            articles.values(),
            key=lambda x: x["distance"]
        )

        return {
            "ids": [[x["id"] for x in selected]],
            "documents": [[x["document"] for x in selected]],
            "metadatas": [[x["metadata"] for x in selected]],
            "distances": [[x["distance"] for x in selected]],
        }


    def _rerank_candidates(
        self,
        results,
    ):
        """
        Compute the final score of each retrieved article
        and sort candidates by decreasing score.
        """

        semantic_scores = self._normalize_distances(
            results["distances"][0]
        )

        candidates = []

        for (
            doc_id,
            document,
            metadata,
            distance,
            semantic_score,
        ) in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            semantic_scores,
        ):

            final_score = self._compute_score(
                semantic_score=semantic_score,
                metadata=metadata,
            )

            candidates.append(
                {
                    "score": final_score,
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return candidates


    def _select_top_k(
        self,
        candidates,
        top_k,
    ):
        """
        Convert ranked candidates into the Chroma format.
        """

        selected = candidates[:top_k]

        return {
            "ids": [[x["id"] for x in selected]],
            "documents": [[x["document"] for x in selected]],
            "metadatas": [[x["metadata"] for x in selected]],
            "distances": [[x["distance"] for x in selected]],
        }


    def retrieve(
        self,
        query,
        top_k=config['retrieval']['top_k']
    ):

        results = self._retrieve_candidates(
            query=query,
            search_k=top_k
        )

        return results


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
        search_k = self._get_search_k(top_k)

        results = self._retrieve_candidates(
            query=query,
            search_k=top_k
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


    def retrieve_diverse_articles(
        self,
        query,
        top_k=config["retrieval"]["top_k"]
    ):
        search_k = self._get_search_k(top_k)

        results = self._retrieve_candidates(
            query=query,
            search_k=top_k
        )

        seen_pmids = set()

        selected_ids = []
        selected_documents = []
        selected_metadatas = []
        selected_distances = []

        for doc_id, doc, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            pmid = metadata["pmid"]

            if pmid in seen_pmids:
                continue

            seen_pmids.add(pmid)

            selected_ids.append(doc_id)
            selected_documents.append(doc)
            selected_metadatas.append(metadata)
            selected_distances.append(distance)

            if len(selected_ids) >= top_k:
                break

        return {
            "ids": [selected_ids],
            "documents": [selected_documents],
            "metadatas": [selected_metadatas],
            "distances": [selected_distances]
        }


    def retrieve_reranked(
        self,
        query,
        top_k=config["retrieval"]["top_k"]
    ):
        """
        Retrieve articles using semantic search followed
        by metadata-based reranking.
        """

        search_k = self._get_search_k(top_k)

        # search
        results = self._retrieve_candidates(
            query=query,
            search_k=search_k,
        )

        # remove duplicates
        results = self._unique_articles(results)

        # Score
        candidates = self._rerank_candidates(
            results
        )

        # output format
        return self._select_top_k(
            candidates,
            top_k,
        )