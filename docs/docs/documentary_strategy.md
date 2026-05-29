# Documentary Strategy

Here, we choose to orient the medical assistant on neuro-oncology and brain imaging.

## Documents sources

We have chosen to focus on PubMed, which is a large database of high-quality scientific articles.

We will use the following data:
- abstracts
- titles
- keywords
- publication year
- journal

We will not use the entire article due to the significant amount of space it requires and the strong strategy it demands. Moreover, there are access rights issues. The above elements are more than sufficient to create a powerful RAG.

## Numbers Of Documents

First, we chose to limit the number of documents to **between 500 and 2,000**. This is sufficient to create a powerful RAG while striking a good balance between quality and performance.

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

First, we will use **"all-MiniLM-L6-v2"** for embedding. It is:
- very fast
- lightweight
- excellent benchmark retrieval
- perfect CPU
- widely used in RAG

## VectorDB

We will use **Chroma** as our VectorDB. It is simple and efficient to use.