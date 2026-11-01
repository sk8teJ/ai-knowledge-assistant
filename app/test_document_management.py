
from vector_store import (
    add_chunks,
    list_documents,
    delete_document
)


TEST_ID = "document_management_test"

# Создаём тестовый документ
add_chunks(
    chunks=[
        "Первый тестовый фрагмент документа.",
        "Второй тестовый фрагмент документа.",
        "Третий тестовый фрагмент документа."
    ],
    source="Тест управления.txt",
    document_id=TEST_ID
)

# Получаем список документов
documents = list_documents()

print("Список документов:")

for document in documents:
    print(document)

# Проверяем наличие нашего документа
test_document = next(
    (
        document
        for document in documents
        if document["document_id"] == TEST_ID
    ),
    None
)

if test_document is None:
    print("ОШИБКА: документ не найден.")

else:
    print("\nКоличество фрагментов:", test_document["chunks_count"])

    # Удаляем тестовый документ
    delete_document(TEST_ID)

    print("\nТестовый документ удалён.")

    # Проверяем, что его больше нет
    remaining_documents = list_documents()

    exists = any(
        document["document_id"] == TEST_ID
        for document in remaining_documents
    )

    if not exists:
        print("УСПЕХ: управление документами работает!")
    else:
        print("ОШИБКА: документ не удалился.")
