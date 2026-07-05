from app.ingestion.pdf_loader import PDFLoader


loader = PDFLoader()

pages = loader.load(r"data\documents\Python_tutorial.pdf")

print(f"Extracted pages: {len(pages)}")

for page in pages[:50]:
    print("\n--- PAGE ---")
    print(page["metadata"])
    print(page["text"][:1000])