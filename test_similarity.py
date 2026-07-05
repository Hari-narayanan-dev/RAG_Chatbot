import numpy as np
from app.ingestion.embedder import Embedder


embedder = Embedder()


sentences = [
    "Employees receive paid vacation days.",
    "Staff members are entitled to annual leave.",
    "Python is used for backend development.",
]


embeddings = embedder.model.encode(
    sentences,
    convert_to_numpy=True,
    normalize_embeddings=True,
)


similarity_1 = np.dot(
    embeddings[0],
    embeddings[1],
)

similarity_2 = np.dot(
    embeddings[0],
    embeddings[2],
)


print(
    "Vacation vs annual leave:",
    similarity_1,
)

print(
    "Vacation vs Python:",
    similarity_2,
)