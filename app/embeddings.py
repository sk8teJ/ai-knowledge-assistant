import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

_client = None


def _get_client():
    """Create the Gemini client only when an API call is needed."""
    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Не найден GEMINI_API_KEY. Добавьте ключ в файл .env."
            )

        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    return _client


def get_friendly_error(error: Exception) -> str:
    """Convert Gemini API errors into user-friendly messages."""
    error_text = str(error).lower()

    if "429" in error_text or "resource_exhausted" in error_text:
        return (
            "Превышен лимит запросов Gemini API. "
            "Подождите немного и проверьте ограничения своей квоты."
        )

    if "503" in error_text or "unavailable" in error_text:
        return (
            "Сервис Gemini временно перегружен или недоступен. "
            "Попробуйте повторить запрос позже."
        )

    if "401" in error_text or "403" in error_text:
        return "Нет доступа к Gemini API. Проверьте API-ключ и права доступа."

    if any(token in error_text for token in ("connection", "timeout", "network")):
        return "Не удалось подключиться к Gemini API. Проверьте интернет-соединение."

    return (
        "Не удалось получить embeddings от Gemini API. "
        f"Техническая информация: {error}"
    )


def get_embedding(text: str) -> list[float]:
    """Convert non-empty text into a Gemini embedding vector."""
    if not text or not text.strip():
        raise ValueError("Нельзя создать вектор из пустого текста")

    try:
        response = _get_client().models.embed_content(
            model="gemini-embedding-2",
            contents=text,
        )

        if not response.embeddings or not response.embeddings[0].values:
            raise RuntimeError("Gemini вернул пустой embedding.")

        return response.embeddings[0].values

    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError(get_friendly_error(error)) from error
