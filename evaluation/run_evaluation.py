import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"
REPORT_FILE = PROJECT_ROOT / "evaluation" / "generation_report.json"

REFUSAL_MARKERS = [
    "нет информации",
    "не найдено информации",
    "не найдено достаточно подходящей информации",
    "не содержится информации",
    "отсутствует информация",
    "не указано",
    "не могу определить",
    "в документах нет",
    "в предоставленных документах нет",
    "недостаточно информации"
]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def check_answer(answer: str, expected_answers: list[str], should_answer: bool) -> bool:
    """Basic lexical check; this does not prove semantic correctness."""
    normalized_answer = normalize_text(answer)

    if should_answer:
        if not expected_answers:
            return False
        return any(normalize_text(expected) in normalized_answer for expected in expected_answers)

    return any(marker in normalized_answer for marker in REFUSAL_MARKERS)


def run_evaluation():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("Набор questions.json пуст.")

    for item in questions:
        if not item.get("question") or "should_answer" not in item:
            raise ValueError(f"Некорректный тест: {item}")
        if item["should_answer"] and not item.get("expected_answers"):
            raise ValueError(f"Для отвечаемого вопроса не заданы expected_answers: {item}")

    # Import only for a live run; unit tests can import check_answer offline.
    sys.path.insert(0, str(PROJECT_ROOT / "app"))
    from rag_pipeline import answer_question

    results = []
    passed = 0
    errors = 0

    print("\nТЕСТИРОВАНИЕ AI KNOWLEDGE ASSISTANT")
    print("=" * 60)

    for index, test in enumerate(questions, start=1):
        question = test["question"]
        print(f"\nТест {index}/{len(questions)} [{test.get('id', 'без ID')}]")
        print(f"Вопрос: {question}")

        try:
            answer, sources = answer_question(question)
            lexical_pass = check_answer(
                answer,
                test.get("expected_answers", []),
                test["should_answer"]
            )
            source_pass = bool(sources) if test["should_answer"] else True
            success = lexical_pass and source_pass
            status = "PASS" if success else "FAIL"
            passed += int(success)
            print(f"Результат: {status}")
            print(f"Ответ: {answer}")
            print(f"Найдено источников: {len(sources)}")
        except Exception as error:
            status = "ERROR"
            errors += 1
            answer = ""
            sources = []
            lexical_pass = False
            source_pass = False
            print(f"Результат: ERROR; {error}")

        results.append({
            "id": test.get("id"),
            "question": question,
            "should_answer": test["should_answer"],
            "status": status,
            "lexical_check": lexical_pass,
            "source_check": source_pass,
            "sources_count": len(sources),
            "answer": answer,
            "error": str(error) if status == "ERROR" else None
        })

    total = len(questions)
    report = {
        "evaluation_date": datetime.now().isoformat(timespec="seconds"),
        "total_questions": total,
        "passed": passed,
        "failed": total - passed - errors,
        "errors": errors,
        "accuracy": round(passed / total, 4) if total else 0,
        "evaluation_note": (
            "Автоматическая лексическая проверка ожидаемых фраз и маркеров отказа. "
            "Она не заменяет экспертную оценку смысловой точности и подтверждения каждого утверждения источником."
        ),
        "results": results
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    print("\n" + "=" * 60)
    print(f"Всего тестов: {total}")
    print(f"Успешно: {passed}")
    print(f"Не пройдено: {report['failed']}")
    print(f"Ошибок выполнения: {errors}")
    print(f"Успешность: {report['accuracy'] * 100:.1f}%")
    print(f"Отчёт сохранён: {REPORT_FILE}")


if __name__ == "__main__":
    run_evaluation()
