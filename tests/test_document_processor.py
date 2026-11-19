import unittest
from io import BytesIO

from docx import Document

from app.document_processor import extract_text


def make_docx():
    document = Document()
    document.add_paragraph("Introduction")

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Revenue"
    table.cell(1, 1).text = "120"

    document.add_paragraph("Conclusion")
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class DocumentProcessorTests(unittest.TestCase):
    def test_docx_extracts_table_cells_in_document_order(self):
        text = extract_text("report.DOCX", make_docx())

        self.assertEqual(
            text.splitlines(),
            [
                "Introduction",
                "Metric | Value",
                "Revenue | 120",
                "Conclusion",
            ],
        )

    def test_docx_ignores_empty_paragraphs_and_table_rows(self):
        document = Document()
        document.add_paragraph("   ")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = ""
        table.cell(0, 1).text = "   "
        buffer = BytesIO()
        document.save(buffer)

        self.assertEqual(extract_text("empty.docx", buffer.getvalue()), "")

    def test_txt_decodes_utf8_and_replaces_invalid_bytes(self):
        self.assertEqual(extract_text("notes.txt", b"hello \xff"), "hello \ufffd")

    def test_unsupported_extension_raises_value_error(self):
        with self.assertRaises(ValueError):
            extract_text("notes.rtf", b"content")


if __name__ == "__main__":
    unittest.main()