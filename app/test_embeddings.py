from embeddings import get_embedding


text = "Ежегодный отпуск составляет 28 календарных дней."

vector = get_embedding(text)

print("Текст успешно преобразован в вектор!")
print("Количество чисел:", len(vector))
print("Первые 5 чисел:", vector[:5])