from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_splitter import TextSplitter
from app.ingestion.embedder import Embedder


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

embeddings = embedder.embed_documents(chunks)


print(f"Embedding shape: {embeddings.shape}")

print("\nFirst chunk:")
print(chunks[0]["text"][:300])

print("\nFirst embedding:")
print(embeddings[0][:10])