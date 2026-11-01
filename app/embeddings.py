
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Загружаем настройки из .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Не найден GEMINI_API_KEY в файле .env"
    )


# Тайм-аут запроса — 30 секунд
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=30000,
        retry_options=types.HttpRetryOptions(
            attempts=1
        )
    )
)


def get_friendly_error(error: Exception) -> str:
    """
    Преобразует технические ошибки Gemini
    в понятные сообщения.
    """

    error_text = str(error).lower()

    if "429" in error_text or "resource_exhausted" in error_text:
        return (
            "Превышен лимит запросов Gemini API. "
            "Подождите немного и попробуйте снова. "
            "Также проверьте ограничения своей квоты."
        )

    if "503" in error_text or "unavailable" in error_text:
        return (
            "Сервис Gemini временно перегружен или недоступен. "
            "Попробуйте повторить запрос через некоторое время."
        )

    if "401" in error_text or "403" in error_text:
        return (
            "Ошибка доступа к Gemini API. "
            "Проверьте API-ключ и настройки доступа."
        )

    if (
        "connection" in error_text
        or "timeout" in error_text
        or "network" in error_text
    ):
        return (
            "Не удалось подключиться к Gemini API. "
            "Проверьте интернет-соединение."
        )

    return (
        "Не удалось получить embeddings от Gemini API. "
        "Попробуйте повторить операцию. "
        f"Техническая информация: {error}"
    )


def get_embedding(text: str) -> list[float]:
    """
    Преобразует текст в числовой вектор
    с помощью Gemini Embeddings.
    """

    if not text.strip():
        raise ValueError(
            "Нельзя создать вектор из пустого текста"
        )

    try:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text
        )

        return response.embeddings[0].values

    except Exception as error:
        friendly_message = get_friendly_error(error)

        raise RuntimeError(friendly_message) from error
