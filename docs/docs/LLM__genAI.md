# LLM and GenAI

## Model choice

Given my computer's performance and the time required for the initial run, we opted to combine **Ollama** and **Mistral** locally.

We chose Mistral because it is open source and has good performance, although it is currently lower than that of main models such as OpenAI or Anthropics.

## Prompt Structure

> You are a medical assistant specialized in neuro-oncology.
> 
> Answer the question using ONLY the provided context.
> 
> If the answer is not contained in the context,
> say that the information is not available.
> 
> Context:
> {context}
> 
> Question:
> {question}
> 
> Answer:

For now, we will limit the context to the RAG documents. Initially, we prefer an incomplete answer to an invented one.

*Following an evaluation and benchmarking, it appears that "ollama.chat" takes a long time to start in Python. This may be due to a lack of compatibility with Python 3.12. For now, the stream mode seems to be more efficient.*

## Sources

Currently, sources are shown at the end of the answer rather than being linked to the citation. This is a choice made to simplify the work for the first version.

## Evaluation

First, we manually evaluate a set of simple questions. Later, we will automate the process using JSON and more detailed questions.

## First Version
### Retrieval

Identified limitations:
- Chunks that are too specialized.
- Chunks at the wrong level
- Diversification issues
