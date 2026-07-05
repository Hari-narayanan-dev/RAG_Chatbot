
class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[dict]) -> list[dict]:
        chunks = []
        chunk_id = 0

        for document in documents:
            text = document["text"]
            words = text.split()

            step_size = self.chunk_size - self.chunk_overlap

            for start in range(0, len(words), step_size):
                end = start + self.chunk_size

                chunk_words = words[start:end]

                if not chunk_words:
                    continue

                chunk_text = " ".join(chunk_words)

                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **document["metadata"],
                            "chunk_id": chunk_id,
                        },
                    }
                )

                chunk_id += 1

        return chunks