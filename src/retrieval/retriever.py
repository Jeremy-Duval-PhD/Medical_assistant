import chromadb
from sentence_transformers import SentenceTransformer

from datetime import datetime

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
        processed_query,
    ):
        """
        Compute the embedding of an already processed query.
        """

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


    def _semantic_search(
        self,
        processed_query,
        search_k,
    ):
        """
        Retrieve the top semantic candidates from ChromaDB.
        """

        query_embedding = self._get_query_embedding(
            processed_query
        )

        return self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=search_k,
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


    def _semantic_score(
        self,
        semantic_score,
    ):
        """
        Return normalized semantic similarity.
        """

        return semantic_score


    def _keyword_score(
        self,
        query_keywords,
        metadata,
    ):
        """
        Compute keyword overlap score.
        """

        article_keywords = set(
            metadata.get(
                "keywords",
                []
            )
        )

        if not article_keywords:
            return None

        overlap = len(
            query_keywords &
            article_keywords
        )

        return overlap / len(query_keywords)


    def _mesh_score(
        self,
        query_keywords,
        metadata,
    ):
        """
        Compute MeSH overlap score.
        """

        mesh_terms = set(
            metadata.get(
                "mesh_terms",
                []
            )
        )

        if not mesh_terms:
            return None

        overlap = len(
            query_keywords &
            mesh_terms
        )

        return overlap / len(query_keywords)


    def _retrieval_score(
        self,
        semantic_score,
        query_keywords,
        metadata,
    ):
        """
        Compute retrieval score from semantic similarity,
        keywords and MeSH terms.
        """

        scores = []

        scores.append(
            (
                self._semantic_score(
                    semantic_score
                ),
                config["retrieval"]["semantic_weight"],
            )
        )

        keyword_score = self._keyword_score(
            query_keywords,
            metadata,
        )

        if keyword_score is not None:

            scores.append(
                (
                    keyword_score,
                    config["retrieval"]["keyword_weight"],
                )
            )

        mesh_score = self._mesh_score(
            query_keywords,
            metadata,
        )

        if mesh_score is not None:

            scores.append(
                (
                    mesh_score,
                    config["retrieval"]["mesh_weight"],
                )
            )

        total_weight = sum(
            weight
            for _, weight in scores
        )

        # Re normalize score if keywords or mesh are empty
        return sum(
            score * weight
            for score, weight in scores
        ) / total_weight


    def _bonus_score(
        self,
        metadata,
    ):
        """
        Compute metadata bonus.
        """

        evidence_bonus = (
            metadata.get(
                "evidence_level",
                0,
            )
            * config["retrieval"]["reranking"]["evidence_weight"]
        )

        current_year = datetime.now().year

        age = max(
            0,
            current_year
            - metadata.get(
                "year",
                current_year,
            )
        )

        recency_bonus = (
            1 / (1 + age)
        ) * config["retrieval"]["reranking"]["recency_weight"]

        return (
            evidence_bonus
            + recency_bonus
        )


    def _compute_score(
        self,
        semantic_score,
        query_keywords,
        metadata,
    ):
        """
        Compute final article score.
        """

        retrieval_score = self._retrieval_score(
            semantic_score=semantic_score,
            query_keywords=query_keywords,
            metadata=metadata,
        )

        bonus = self._bonus_score(
            metadata
        )

        return retrieval_score + bonus


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
        query_keywords,
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
                query_keywords=query_keywords,
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
        top_k=config["retrieval"]["top_k"]
    ):

        processed_query = clean_query(query)

        return self._semantic_search(
            processed_query=processed_query,
            search_k=top_k,
        )


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

        processed_query = clean_query(query)
        results = self._semantic_search(
            processed_query=processed_query,
            search_k=search_k,
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
        processed_query = clean_query(query)

        results = self._semantic_search(
            processed_query=processed_query,
            search_k=search_k,
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

        # Preprocess query once
        processed_query = clean_query(query)

        query_keywords = set(
            processed_query.split()
        )

        # Semantic retrieval
        results = self._semantic_search(
            processed_query=processed_query,
            search_k=search_k,
        )

        # Keep one chunk per article
        results = self._unique_articles(results)

        # Metadata reranking
        candidates = self._rerank_candidates(
            query_keywords=query_keywords,
            results=results,
        )

        return self._select_top_k(
            candidates,
            top_k,
        )