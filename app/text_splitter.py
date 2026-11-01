def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[str]:

    """
    Разбивает текст на небольшие фрагменты.
    """

    if not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks