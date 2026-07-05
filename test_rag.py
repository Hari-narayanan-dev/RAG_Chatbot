from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_splitter import TextSplitter
from app.ingestion.embedder import Embedder

from app.retrieval.vector_store import VectorStore
from app.retrieval.prompt_builder import PromptBuilder

from app.llm.generator import Generator


# ---------------------------
# INGESTION PIPELINE
# ---------------------------

loader = PDFLoader()

documents = loader.load(
    r"C:\Users\hp\Desktop\Resume_projects\RAG_Chatbot\data\documents\Python_tutorial.pdf"
)

print(f"Pages extracted: {len(documents)}")


splitter = TextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

chunks = splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


embedder = Embedder()

document_embeddings = embedder.embed_documents(
    chunks
)

print(
    f"Embedding shape: "
    f"{document_embeddings.shape}"
)


vector_store = VectorStore(
    dimension=document_embeddings.shape[1]
)

vector_store.add(
    embeddings=document_embeddings,
    documents=chunks,
)


# ---------------------------
# QUERY PIPELINE
# ---------------------------

question = "what is python string?"


query_embedding = embedder.embed_query(
    question
)


retrieved_chunks = vector_store.search(
    query_embedding=query_embedding,
    top_k=3,
)


print("\nRETRIEVED SOURCES:")

for result in retrieved_chunks:
    print(
        result["metadata"],
        f"score={result['score']:.4f}",
    )


# ---------------------------
# AUGMENTATION
# ---------------------------

prompt_builder = PromptBuilder()

prompt = prompt_builder.build(
    question=question,
    retrieved_chunks=retrieved_chunks,
)


# ---------------------------
# GENERATION
# ---------------------------

generator = Generator()

answer = generator.generate(prompt)


print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)