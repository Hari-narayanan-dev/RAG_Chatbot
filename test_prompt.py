from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_splitter import TextSplitter
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.prompt_builder import PromptBuilder


# 1. Load document
loader = PDFLoader()

documents = loader.load(
    r"C:\Users\hp\Desktop\Resume_projects\RAG_Chatbot\data\documents\Python_tutorial.pdf"
)


# 2. Split document
splitter = TextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

chunks = splitter.split_documents(documents)


# 3. Generate document embeddings
embedder = Embedder()

embeddings = embedder.embed_documents(chunks)


# 4. Create and populate vector store
vector_store = VectorStore(
    dimension=embeddings.shape[1]
)

vector_store.add(
    embeddings=embeddings,
    documents=chunks,
)


# 5. Ask a question
question = "What is this document mainly about?"


# 6. Embed the question
query_embedding = embedder.embed_query(
    question
)


# 7. Retrieve relevant chunks
results = vector_store.search(
    query_embedding=query_embedding,
    top_k=3,
)


# 8. Build augmented prompt
prompt_builder = PromptBuilder()

prompt = prompt_builder.build(
    question=question,
    retrieved_chunks=results,
)


# 9. Inspect the exact LLM input
print(prompt)