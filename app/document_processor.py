from io import BytesIO
from pathlib import Path

import fitz
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


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
        text_parts = []

        # Сохраняем порядок абзацев и таблиц в документе.
        for element in document.element.body.iterchildren():
            if element.tag == qn("w:p"):
                paragraph = Paragraph(element, document)
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)

            elif element.tag == qn("w:tbl"):
                table = Table(element, document)
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    if any(cells):
                        text_parts.append(" | ".join(cells))

        return "\n".join(text_parts)

    # Чтение обычного текстового файла
    elif extension == ".txt":
        return file_bytes.decode("utf-8", errors="replace")

    else:
        raise ValueError("Неподдерживаемый формат файла")