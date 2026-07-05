from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_splitter import TextSplitter


loader = PDFLoader()

documents = loader.load(
    r"data\documents\Python_tutorial.pdf"
)

print(f"Pages extracted: {len(documents)}")


splitter = TextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

chunks = splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


for chunk in chunks[50:55]:
    print("\n" + "=" * 60)

    print("METADATA:")
    print(chunk["metadata"])

    print("\nTEXT:")
    print(chunk["text"])