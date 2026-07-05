from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_splitter import TextSplitter
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore


# 1. Load PDF
loader = PDFLoader()

documents = loader.load(
    r"C:\Users\hp\Desktop\Resume_projects\RAG_Chatbot\data\documents\Python_tutorial.pdf"
)

print(f"Pages extracted: {len(documents)}")


# 2. Create chunks
splitter = TextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

chunks = splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# 3. Generate document embeddings
embedder = Embedder()

embeddings = embedder.embed_documents(chunks)

print(f"Embedding shape: {embeddings.shape}")


# 4. Create vector index
dimension = embeddings.shape[1]

vector_store = VectorStore(
    dimension=dimension
)


# 5. Add vectors and chunks
vector_store.add(
    embeddings=embeddings,
    documents=chunks,
)


# 6. User question
query = "What is string?"


# 7. Convert question into embedding
query_embedding = embedder.embed_query(query)


# 8. Semantic search
results = vector_store.search(
    query_embedding=query_embedding,
    top_k=3,
)


# 9. Inspect results
print(f"\nQUERY: {query}")


for rank, result in enumerate(
    results,
    start=1,
):
    print("\n" + "=" * 70)

    print(f"RANK: {rank}")
    print(f"SCORE: {result['score']:.4f}")
    print(f"METADATA: {result['metadata']}")

    print("\nTEXT:")
    print(result["text"])