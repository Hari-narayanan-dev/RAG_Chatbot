from pathlib import Path
import pymupdf



from pathlib import Path
import pymupdf


class PDFLoader:
    """Extract text and page metadata from text-based PDF documents."""

    def load(self, file_path: str) -> list[dict]:
        pdf_path = Path(file_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, received: {pdf_path.suffix}")

        pages = []

        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if not text:
                    continue

                pages.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_number,
                        },
                    }
                )

        return pages