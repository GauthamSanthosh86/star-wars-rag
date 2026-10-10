import requests
import re
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

GEMINI_MODEL = "gemini-3.5-flash-lite"  # fallback: "gemini-3.1-flash-lite"
PERSIST_PATH = "./qdrant_db"
COLLECTION_NAME = "star_wars_scripts"

# Local, free embeddings (384-dim). First run downloads the model.
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0) # termperature argument doesn't work with gemini-flash-lite
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)

def load_star_wars_script(url, movie_title):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    pre = soup.find("pre")
    if pre is None:
        raise ValueError(f"No <pre> block found for {movie_title}")
    return Document(page_content=pre.get_text(), metadata={"title": movie_title})


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{d.metadata['title']}]\n{d.page_content}" for d in docs
    )

def scene_heading(text):
    """Return the first INT./EXT. line in chunk or a fallback"""
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^(INT\.|EXT\.|INT/EXT)", line):
            return line
    return "(Continues a scene)"

def print_sources(docs):
    seen = set()
    for d in docs:
        key = (d.metadata["title"], d.metadata.get("start_index"))
        if key in seen:
            continue
        seen.add(key)
        snippet = " ".join(d.page_content.split())[:140]
        print(f"  - {d.metadata['title']} | {scene_heading(d.page_content)}")
        print(f"    \"{snippet}...\"")

def main():
    client = QdrantClient(path=PERSIST_PATH)

    if client.collection_exists(COLLECTION_NAME):
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
        print("Collection already exists. Loaded existing collection.")
    else:
        client.close()  # release the lock so from_documents can open the path

        star_wars_scripts = [
            {
                "title": "Star Wars: A New Hope",
                "url": "https://www.imsdb.com/scripts/Star-Wars-A-New-Hope.html",
            },
            {
                "title": "Star Wars: The Empire Strikes Back",
                "url": "https://www.imsdb.com/scripts/Star-Wars-The-Empire-Strikes-Back.html",
            },
            {
                "title": "Star Wars: Return of the Jedi",
                "url": "https://www.imsdb.com/scripts/Star-Wars-Return-of-the-Jedi.html",
            },
        ]

        # Smaller chunks: bge-small reads ~512 tokens
        script_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            add_start_index=True,
            separators=["\nINT", "\nEXT", "\n\n", "\n", " ", ""],
        )

        all_chunks = []
        for script in star_wars_scripts:
            doc = load_star_wars_script(script["url"], script["title"])
            chunks = script_splitter.split_documents([doc])
            all_chunks.extend(chunks)
            print(f"Loaded and split {script['title']} into {len(chunks)} chunks.")

        vectorstore = QdrantVectorStore.from_documents(
            all_chunks,
            embedding=embeddings,
            path=PERSIST_PATH,
            collection_name=COLLECTION_NAME,
        )
        print(f"Stored {len(all_chunks)} chunks in Qdrant.")

    # retriever = vectorstore.as_retriever(search_kwargs={"k": 8}) old one
    retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs ={"k": 15, "fetch_k": 50, "lambda_mult": 0.8} 
    )

    template = """You are a Star Wars Movie Script Expert. Use ONLY the following script excerpts to answer.
    If answer is partly contained. Provide best possible answer based on text in the context.
    If the excerpts only partly answer the question,
    say what is covered and what is missing. 
    Never combine numbers or facts from different excerpts unless the text clearly links them.
If the answer isn't in the context, say "There is no information about this in the original Star Wars Scripts."

Write the answer as one short paragraph of plain prose. Do not use bullet points, bold text, or headings.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        | RunnablePassthrough.assign(answer=answer_chain)
    )

    print("\n--- The Star Wars Movie Expert is ready to answer your questions ---")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            result = rag_chain.invoke(query)
            print(f"\nStar Wars Movie Expert: {result['answer']}")
            print_sources(result["docs"])
        except Exception as e:
            print(f"\nError: {str(e)[:200]}")


if __name__ == "__main__":
    main()