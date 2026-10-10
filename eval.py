"""Retrieval evaluation for the Star Wars RAG project.

For each test question we check whether the retrieved chunks contain a
"must-find" phrase from the real script. No LLM calls, so no API quota used.

Run with main.py CLOSED (Qdrant local mode locks the folder):
    python eval.py
"""
import re
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_PATH = "./qdrant_db"
COLLECTION_NAME = "star_wars_scripts"

# "must" is a phrase that should appear in a correct chunk (case, punctuation
# and line breaks are ignored). "movie" restricts the match to one film
# (substring of the title); use None to accept any film.
# These phrases are written from memory: the script's corpus check below
# flags any that don't exist in your data, so fix those before trusting results.
TESTS = [
    {"q": "What does Obi-Wan say the Force is?",
     "must": "energy field created by all living things", "movie": "New Hope"},
    {"q": "How much does Han ask for in advance to take them to Alderaan?",
     "must": "Ten thousand in advance", "movie": "New Hope"},
    {"q": "What does Han demand up front for the trip?",
     "must": "Ten thousand in advance", "movie": "New Hope"},
    {"q": "What does Han boast about the Millennium Falcon?",
     "must": "Kessel run in less than twelve parsecs", "movie": "New Hope"},
    {"q": "What does Leia say in her message to Obi-Wan?",
     "must": "You're my only hope", "movie": "New Hope"},
    {"q": "What does Obi-Wan tell the stormtroopers about the droids?",
     "must": "aren't the droids you're looking for", "movie": "New Hope"},
    {"q": "What does Luke say when Ben mentions going to Alderaan?",
     "must": "I'm not going to Alderaan", "movie": "New Hope"},
    {"q": "Who says may the Force be with you?",
     "must": "May the Force be with you", "movie": "New Hope"},
    {"q": "What does Vader reveal to Luke about his father?",
     "must": "I am your father", "movie": "Empire"},
    {"q": "What does Yoda say about trying?",
     "must": "There is no try", "movie": "Empire"},
    {"q": "What does Yoda say about size?",
     "must": "Size matters not", "movie": "Empire"},
    {"q": "What does Han say about the odds?",
     "must": "Never tell me the odds", "movie": "Empire"},
    {"q": "What does Admiral Ackbar warn about?",
     "must": "It's a trap", "movie": "Jedi"},
    {"q": "Who has a bad feeling about this?",
     "must": "bad feeling about this", "movie": None},
]

CONFIGS = {
    "similarity k=8":  dict(search_type="similarity", search_kwargs={"k": 8}),
    "similarity k=15": dict(search_type="similarity", search_kwargs={"k": 15}),
    "mmr k=15 l=0.8":  dict(search_type="mmr",
                            search_kwargs={"k": 15, "fetch_k": 50, "lambda_mult": 0.8}),
    "mmr k=15 l=0.5":  dict(search_type="mmr",
                            search_kwargs={"k": 15, "fetch_k": 50, "lambda_mult": 0.5}),
}


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def matches(doc_text, doc_title, test):
    if test["movie"] and test["movie"].lower() not in doc_title.lower():
        return False
    return norm(test["must"]) in norm(doc_text)


def load_all_chunks(client):
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(
            COLLECTION_NAME, limit=256, offset=offset, with_payload=True
        )
        for p in points:
            payload = p.payload or {}
            meta = payload.get("metadata", {})
            chunks.append((payload.get("page_content", ""), meta.get("title", "")))
        if offset is None:
            break
    return chunks


def main():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    client = QdrantClient(path=PERSIST_PATH)
    vs = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings
    )

    # Step 1: corpus check. Does each must-find phrase exist in some chunk?
    chunks = load_all_chunks(client)
    print(f"Loaded {len(chunks)} chunks from the database.\n")
    valid = []
    print("Corpus check (phrase exists in at least one chunk):")
    for t in TESTS:
        found = any(matches(text, title, t) for text, title in chunks)
        print(f"  [{'ok' if found else 'MISSING'}] {t['must']!r}")
        if found:
            valid.append(t)
    skipped = len(TESTS) - len(valid)
    if skipped:
        print(f"\n{skipped} phrase(s) not found: fix the wording against the real "
              "script (or the phrase may be split across a chunk boundary). "
              "They are excluded below.")
    if not valid:
        return

    # Step 2: retrieval hit rate per config
    print(f"\nRetrieval hit rate on {len(valid)} valid questions:\n")
    print(f"{'config':<18} {'hits':>6} {'rate':>7}")
    details = {}
    for name, cfg in CONFIGS.items():
        retriever = vs.as_retriever(**cfg)
        hits, misses = 0, []
        for t in valid:
            docs = retriever.invoke(t["q"])
            if any(matches(d.page_content, d.metadata.get("title", ""), t) for d in docs):
                hits += 1
            else:
                misses.append(t["q"])
        details[name] = misses
        print(f"{name:<18} {hits:>3}/{len(valid):<3} {hits / len(valid):>6.0%}")

    print("\nMissed questions per config:")
    for name, misses in details.items():
        print(f"  {name}:")
        for q in misses or ["(none)"]:
            print(f"    - {q}")


if __name__ == "__main__":
    main()