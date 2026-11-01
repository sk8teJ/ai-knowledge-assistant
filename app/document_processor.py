from io import BytesIO
from pathlib import Path

import fitz
from docx import Document


def extract_text(file_name: str, file_bytes: bytes) -> str:
    """
    Извлекает текст из PDF, DOCX или TXT.
    """

    extension = Path(file_name).suffix.lower()

    # Чтение PDF
    if extension == ".pdf":
        text_parts = []

        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            for page in pdf:
                text_parts.append(page.get_text())

        return "\n".join(text_parts)

    # Чтение Word
    elif extension == ".docx":
        document = Document(BytesIO(file_bytes))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)

    # Чтение обычного текстового файла
    elif extension == ".txt":
        return file_bytes.decode("utf-8", errors="replace")

    else:
        raise ValueError("Неподдерживаемый формат файла")