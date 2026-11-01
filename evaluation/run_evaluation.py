
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "app"))

from rag_pipeline import answer_question


QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"

REFUSAL_MARKERS = [
    "нет информации",
    "не найдено информации",
    "не найдено достаточно подходящей информации",
    "не содержится информации",
    "отсутствует информация",
    "не указано",
    "не могу определить",
    "в документах нет",
    "в предоставленных документах нет"
]


def normalize_text(text: str) -> str:
    """
    Приводит текст к единому формату.
    """

    return " ".join(text.lower().split())


def check_answer(
    answer: str,
    expected_answers: list[str],
    should_answer: bool
) -> bool:
    """
    Проверяет правильность ответа.
    """

    normalized_answer = normalize_text(answer)

    if should_answer:
        return any(
            normalize_text(expected) in normalized_answer
            for expected in expected_answers
        )

    return any(
        marker in normalized_answer
        for marker in REFUSAL_MARKERS
    )


def run_evaluation():
    """
    Запускает тестирование RAG-системы.
    """

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    total = len(questions)
    passed = 0

    print("\nТЕСТИРОВАНИЕ AI KNOWLEDGE ASSISTANT")
    print("=" * 50)

    for index, test in enumerate(questions, start=1):

        question = test["question"]
        expected_answers = test["expected_answers"]
        should_answer = test["should_answer"]

        print(f"\nТест {index}/{total}")
        print(f"Вопрос: {question}")

        try:
            answer, sources = answer_question(question)

            success = check_answer(
                answer,
                expected_answers,
                should_answer
            )

            if success:
                passed += 1
                status = "PASS"
            else:
                status = "FAIL"

            print(f"Результат: {status}")
            print(f"Ответ: {answer}")
            print(f"Найдено источников: {len(sources)}")

        except Exception as error:
            print("Результат: ERROR")
            print(f"Ошибка: {error}")

    failed = total - passed
    accuracy = passed / total * 100 if total else 0

    print("\n" + "=" * 50)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print(f"Всего тестов: {total}")
    print(f"Успешно: {passed}")
    print(f"Не пройдено: {failed}")
    print(f"Успешность тестирования: {accuracy:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()
