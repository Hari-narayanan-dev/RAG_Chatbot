class PromptBuilder:
    def build(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        context_sections = []

        for rank, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            metadata = chunk["metadata"]

            source = metadata.get(
                "source",
                "Unknown source",
            )

            page = metadata.get(
                "page",
                "Unknown page",
            )

            section = (
                f"[Context {rank}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{chunk['text']}"
            )

            context_sections.append(section)

        context = "\n\n".join(context_sections)

        prompt = f"""
You are a knowledge-base assistant.

Answer the user's question using only the information provided
in the context below.

Instructions:
- Do not use outside knowledge.
- If the answer is not available in the context, say:
  "I could not find the answer in the provided knowledge base."
- Do not invent facts.
- Keep the answer clear and concise.
- Use the source information when referring to evidence.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

        return prompt