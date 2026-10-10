# Star Wars Script Expert (RAG)

A retrieval-augmented question-answering system over the three original Star Wars scripts. Ask a question, get a short answer grounded in the scripts, plus the movie, scene and snippet each answer was drawn from.

Started from Better Stack's "How to Build a RAG System That Actually Works" tutorial. See [What I changed](#what-i-changed) for my additions.

## How it works

1. **Load:** scrape the scripts from IMSDB (A New Hope, The Empire Strikes Back, Return of the Jedi).
2. **Chunk:** split into ~1200-character chunks (150 overlap), preferring scene boundaries (`INT.` / `EXT.`). 677 chunks total.
3. **Embed and store:** `BAAI/bge-small-en-v1.5` (local, free) embeddings stored in a local Qdrant database.
4. **Retrieve:** top 15 chunks by similarity for each question.
5. **Answer:** Gemini answers using only those chunks. The prompt tells it to say so when the context is missing or partial, and not to combine facts across excerpts.
6. **Cite:** print the source movie, scene heading and a snippet for each retrieved chunk.

**Stack:** Python, LangChain, Qdrant (local mode), Hugging Face sentence-transformers, Gemini API.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"    # free key from aistudio.google.com
python main.py
```

The first run downloads the embedding model and builds `./qdrant_db` (a few minutes). Later runs load it directly. Type `exit` to quit.

Example questions:
- What does Obi-Wan tell Luke about the Force?
- How much does Han ask for in advance to take them to Alderaan?
- Who is Rey? (not in the scripts, so it should say there's no information)

## Evaluation

`eval.py` measures **retrieval hit rate**: for each test question, do the retrieved chunks contain a known phrase from the script? It makes no LLM calls. A corpus check first confirms each phrase exists in the data, so a mistyped phrase isn't counted as a failure.

Results on a 13-question test set:

| Retriever | Hit rate |
|---|---|
| Similarity, k=8 | 62% |
| **Similarity, k=15 (used)** | **85%** |
| MMR, k=15, lambda=0.8 | 62% |
| MMR, k=15, lambda=0.5 | 69% |

I expected MMR to help by reducing near-duplicate chunks, but plain similarity with a larger `k` did better on this data, so I switched.

Run it with `python eval.py` (close `main.py` first; local Qdrant allows one process at a time).

## What I changed

- Local embeddings and Gemini instead of OpenAI, so it runs free
- Source citations (movie, scene heading, snippet) on every answer
- Prompt changes: plain-prose answers, honest handling of partial context, no mixing facts across excerpts
- Tested an out-of-scope question and a prompt-injection attempt
- Wrote the retrieval evaluation and used it to choose the retriever

## Limitations

- **Small test set (13 questions).** One question is about 8 points, so treat the numbers as indicative.
- **Retrieval only.** The eval doesn't grade whether Gemini's final answers are correct.
- **Wording-sensitive.** "How much does Han ask for in advance?" retrieves the right scene, but "What does Han demand up front?" can miss it.
- **Answers vary between runs.** The Flash-Lite model ignores the temperature setting.
- **Chunks lack scene context.** Many chunks start mid-scene, so the scene heading isn't in their text.

