import os

from dotenv import load_dotenv
from google import genai


# Загружаем настройки из .env
load_dotenv()

# Получаем API-ключ
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ОШИБКА: API-ключ не найден в файле .env")

else:
    try:
        # Создаём подключение к Gemini
        client = genai.Client(api_key=api_key)

        # Отправляем тестовый запрос
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Ответь на русском: Привет! Ты работаешь?"
        )

        print("Подключение к Gemini успешно!")
        print("Ответ модели:")
        print(response.text)

        client.close()

    except Exception as error:
        print("Не удалось выполнить запрос.")
        print("Тип ошибки:", type(error).__name__)
        print("Описание:", str(error).replace(api_key, "[СКРЫТЫЙ КЛЮЧ]"))