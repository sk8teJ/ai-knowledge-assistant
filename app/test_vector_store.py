from vector_store import add_chunks, search_chunks


# Пример фрагментов документа
chunks = [
    "Ежегодный оплачиваемый отпуск составляет 28 календарных дней.",
    "Рабочий день начинается в 9:00 утра.",
    "Заявление на отпуск необходимо согласовать с руководителем."
]

# Сохраняем фрагменты в базу
count = add_chunks(
    chunks=chunks,
    source="test_document.txt"
)

print("Добавлено фрагментов:", count)

# Ищем информацию по вопросу
results = search_chunks(
    query="Сколько дней длится ежегодный отпуск?",
    n_results=3
)

print("\nРезультаты поиска:")

for result in results:
    print("\nИсточник:", result["source"])
    print("Текст:", result["text"])
    print("Расстояние:", result["distance"])