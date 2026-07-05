import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.documents = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: list[dict],
    ) -> None:
        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents"
            )

        embeddings = embeddings.astype("float32")

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        ).reshape(1, -1)

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            document = self.documents[index]

            results.append(
                {
                    "text": document["text"],
                    "metadata": document["metadata"],
                    "score": float(score),
                }
            )

        return results