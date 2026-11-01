
from vector_store import add_chunks, collection


# Два разных документа с одинаковым названием
SOURCE = "Инструкция.docx"

ID_FIRST = "test_document_first"
ID_SECOND = "test_document_second"

# Удаляем только записи предыдущего теста
for document_id in [ID_FIRST, ID_SECOND]:
    collection.delete(
        where={"document_id": document_id}
    )


print("Загружаем первый документ...")

add_chunks(
    chunks=[
        "Продолжительность отпуска составляет 28 календарных дней."
    ],
    source=SOURCE,
    document_id=ID_FIRST
)


print("Загружаем второй документ...")

add_chunks(
    chunks=[
        "Продолжительность отпуска составляет 35 календарных дней."
    ],
    source=SOURCE,
    document_id=ID_SECOND
)


# Получаем оба документа
result = collection.get(
    where={"source": SOURCE},
    include=["documents", "metadatas"]
)

print("\nНайденные документы:")

for document, metadata in zip(
    result["documents"],
    result["metadatas"]
):
    print("Название:", metadata["source"])
    print("ID:", metadata["document_id"])
    print("Текст:", document)
    print("---")


# Проверяем, что сохранились оба документа
success = (
    ID_FIRST in [m["document_id"] for m in result["metadatas"]]
    and
    ID_SECOND in [m["document_id"] for m in result["metadatas"]]
    and
    len(result["documents"]) == 2
)

# Удаляем только тестовые записи
for document_id in [ID_FIRST, ID_SECOND]:
    collection.delete(
        where={"document_id": document_id}
    )


if success:
    print("\nУСПЕХ: оба документа сохранены независимо!")
else:
    print("\nОШИБКА: документы не прошли проверку.")
