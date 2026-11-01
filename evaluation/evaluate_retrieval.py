
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "app"))

from vector_store import search_chunks


QUESTIONS_FILE = (
    PROJECT_ROOT / "evaluation" / "retrieval_questions.json"
)

REPORT_FILE = (
    PROJECT_ROOT / "evaluation" / "retrieval_report.json"
)

TOP_K = 3


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def run_evaluation():

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    hits = 0
    reciprocal_ranks = []
    test_results = []

    print("\nОЦЕНКА КАЧЕСТВА ПОИСКА")
    print("=" * 50)

    for index, test in enumerate(questions, start=1):

        question = test["question"]

        expected_terms = [
            normalize_text(term)
            for term in test["expected_terms"]
        ]

        print(f"\nТест {index}/{len(questions)}")
        print(f"Вопрос: {question}")

        relevant_rank = None
        error_message = None
        retrieved_count = 0

        try:
            results = search_chunks(
                query=question,
                n_results=TOP_K
            )

            retrieved_count = len(results)

            for rank, chunk in enumerate(results, start=1):

                chunk_text = normalize_text(chunk["text"])

                if any(
                    term in chunk_text
                    for term in expected_terms
                ):
                    relevant_rank = rank
                    break

            if relevant_rank is not None:
                hits += 1
                reciprocal_ranks.append(1 / relevant_rank)

                status = "PASS"

                print("Результат: PASS")
                print(
                    f"Нужный фрагмент найден на позиции: "
                    f"{relevant_rank}"
                )

            else:
                reciprocal_ranks.append(0)
                status = "FAIL"

                print("Результат: FAIL")
                print("Нужный фрагмент не найден в TOP-3")

            print(f"Получено фрагментов: {retrieved_count}")

        except Exception as error:

            reciprocal_ranks.append(0)
            status = "ERROR"
            error_message = str(error)

            print("Результат: ERROR")
            print(f"Ошибка: {error}")

        test_results.append({
            "question": question,
            "status": status,
            "relevant_rank": relevant_rank,
            "retrieved_count": retrieved_count,
            "error": error_message
        })

    total = len(questions)

    hit_at_3 = hits / total if total else 0

    mrr = (
        sum(reciprocal_ranks) / total
        if total else 0
    )

    report = {
        "evaluation_date": datetime.now().isoformat(
            timespec="seconds"
        ),
        "total_questions": total,
        "successful_searches": hits,
        "hit_at_3": round(hit_at_3, 4),
        "mrr": round(mrr, 4),
        "top_k": TOP_K,
        "results": test_results
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 50)
    print("ИТОГИ ОЦЕНКИ ПОИСКА")

    print(f"Всего вопросов: {total}")
    print(f"Найдено нужных фрагментов: {hits}")
    print(f"Hit@3: {hit_at_3 * 100:.1f}%")
    print(f"MRR: {mrr:.3f}")

    print("\nОтчёт сохранён:")
    print(REPORT_FILE)

    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()
