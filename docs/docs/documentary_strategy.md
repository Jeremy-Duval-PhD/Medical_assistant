# Documentary Strategy

Here, we choose to orient the medical assistant on neuro-oncology and brain imaging.

## Documents sources

We have chosen to focus on PubMed, which is a large database of high-quality scientific articles.

We will use the following data:
- abstracts
- titles
- keywords
- publication year
- publication type
- journal
- mesh

We added some features, such as the evidence level, which is defined as follows:
| Publication Type  | Evidence Level |
|:-----|:-----:|
| guideline | 4 |
| meta-analysis | 4 |
| systematic review | 3 |
| review | 2 |
| other | 1 |

We will not use the entire article due to the significant amount of space it requires and the strong strategy it demands. Moreover, there are access rights issues. The above elements are more than sufficient to create a powerful RAG.

## Numbers Of Documents

First, we chose to limit the number of documents to **500**. This is sufficient to create a powerful RAG while striking a good balance between quality and performance.

Second, we improved the query and the document corpus. We kept **3,307** unique documents.

## Data Structure

We will use the following JSON structure, which is a classical and an efficient data structure for:
- filtering
- reranking
- citations
- analytics
- scoring

```JSON
{
  "pmid": "...",
  "title": "...",
  "abstract": "...",
  "year": 2024,
  "journal": "...",
  "keywords": [...],
  "mesh_terms": [...],
  "source": "PubMed"
}
```

## Data Management

We will use [DVC](https://dvc.org/).

## Chunking

We use *NLTK* for sentence tokenization. We will preserve the entire sentence without truncation or overlapping tokens to maintain context.

We will start with the following parameters, which, once again, represent a good compromise.

```Python
chunk_size=500,
overlap_sentences=1
```

For now, we are working with characters and sentences because it is simple to start a new project this way. Then, we will incrementally upgrade the project.

## Embedding 

We use the "SentenceTransformer" from "sentence_transformers" because it is:
- Lightweight
- quick
- widely used for RAG
- excellent rapport qualité/performance
- adapted for laptops

First, we will use the model **"all-MiniLM-L6-v2"** for embedding. It is:
- very fast
- lightweight
- excellent benchmark retrieval
- perfect CPU
- widely used in RAG

## VectorDB And Retriever

We will use **Chroma** as our VectorDB. It is simple and efficient to use.

The retrieval process is currently simple. Later, we can improve it using reranking, metadata filtering (year, preprint...), or hybrid retrieval.

After the baseline system test, we chose to improve the Retriever by:
- PMID diversification.
- Retrieve the entire article instead of only a chunk (informations are frequently splitted into the entire abstract).

# First Run Analysis
## Retrieval Quality

The retrieval system, which is based on embeddings and ChromaDB, generally returns documents that are consistent with the entered queries.

Tests conducted on several medical queries demonstrate that the retrieved chunks are largely relevant to the search intent.

## Document noise

The observed noise level is low.

However, some edge cases remain:

- very general documents
- partially relevant chunks
- occasional retrieval of less specific documents

Future improvements could be made through:

- more advanced query cleaning
- query rewriting
- reranking

## Identified Limitations 
### Document Diversity

Currently, the system does not guarantee that chunks come from different documents.

In some cases, multiple chunks from the same article may dominate the context sent to the LLM.

A strategy to diversify the results may be added at a later date.

### Recency of Knowledge:

The system already utilizes temporal metadata.

A future improvement could involve integrating a hybrid score.

> Semantic relevance + recency of publication

This would prioritize the most recent publications.

## Query preprocessing:

The current preprocessing primarily involves the following:

- normalization
- stopword removal

This approach remains straightforward.

Future improvements could include:

- lemmatization
- extraction of medical entities
- automatic query reformulation using an LLM

## Conclusion:

The retrieval system provides a solid foundation for the initial project version.

The main objectives of the sprint have been achieved:

- cleaned corpus
- consistent chunking
- generated embeddings
- operational ChromaDB vectorization
- functional semantic search

# Retriver upgrade

About 100 chunks are extracted. Only one chunk per article is selected. A hybrid retriever is then used to rank the chunks by relevance: 
- The retrieval score: composed primarily of the semantic score (derived from the embedding), the proportion of keywords, and the proportion of mesh. If no keywords and/or no mesh exist, the weights (see the configuration file) are renormalized to avoid selection bias.
- The bonus score: heavily weighted toward the type of article (level of evidence) and slightly weighted toward the year of publication.