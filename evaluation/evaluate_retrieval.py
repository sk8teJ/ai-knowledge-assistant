import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "retrieval_questions.json"
REPORT_FILE = PROJECT_ROOT / "evaluation" / "retrieval_report.json"
TOP_K = 3


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def calculate_metrics(results: list[dict], top_k: int = TOP_K) -> dict:
    """Calculate ranking metrics from per-question relevant ranks."""
    total = len(results)
    if not total:
        return {"total_questions": 0, "hit_at_1": 0.0, f"hit_at_{top_k}": 0.0, "mrr": 0.0}

    ranks = [item.get("relevant_rank") for item in results]
    hit_at_1 = sum(rank == 1 for rank in ranks) / total
    hit_at_k = sum(rank is not None and rank <= top_k for rank in ranks) / total
    mrr = sum((1 / rank) if rank else 0 for rank in ranks) / total

    return {
        "total_questions": total,
        "hit_at_1": round(hit_at_1, 4),
        f"hit_at_{top_k}": round(hit_at_k, 4),
        "mrr": round(mrr, 4)
    }


def run_evaluation():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("Набор retrieval_questions.json пуст.")

    for item in questions:
        if not item.get("question") or not item.get("expected_terms"):
            raise ValueError(f"Некорректный тест: {item}")

    # Import only when the live evaluation is started. This keeps metric
    # helpers importable in offline unit tests without a Gemini API key.
    sys.path.insert(0, str(PROJECT_ROOT / "app"))
    from vector_store import search_chunks

    test_results = []
    print("\nОЦЕНКА КАЧЕСТВА ПОИСКА")
    print("=" * 60)

    for index, test in enumerate(questions, start=1):
        question = test["question"]
        expected_terms = [normalize_text(term) for term in test["expected_terms"]]
        relevant_rank = None
        retrieved_count = 0
        error_message = None
        matched_terms = []

        print(f"\nТест {index}/{len(questions)}: {question}")

        try:
            chunks = search_chunks(query=question, n_results=TOP_K)
            retrieved_count = len(chunks)

            for rank, chunk in enumerate(chunks, start=1):
                chunk_text = normalize_text(chunk["text"])
                matches = [term for term in expected_terms if term in chunk_text]
                if matches:
                    relevant_rank = rank
                    matched_terms = matches
                    break

            status = "PASS" if relevant_rank is not None else "FAIL"
            print(f"Результат: {status}; позиция: {relevant_rank}")
        except Exception as error:
            status = "ERROR"
            error_message = str(error)
            print(f"Результат: ERROR; {error_message}")

        test_results.append({
            "id": test.get("id"),
            "fact_id": test.get("fact_id"),
            "question": question,
            "status": status,
            "relevant_rank": relevant_rank,
            "retrieved_count": retrieved_count,
            "matched_terms": matched_terms,
            "error": error_message
        })

    metrics = calculate_metrics(test_results, TOP_K)
    fact_results = {}
    for fact_id in sorted({item.get("fact_id") for item in questions if item.get("fact_id")}):
        subset = [result for result in test_results if result.get("fact_id") == fact_id]
        fact_results[fact_id] = calculate_metrics(subset, TOP_K)

    report = {
        "evaluation_date": datetime.now().isoformat(timespec="seconds"),
        "top_k": TOP_K,
        **metrics,
        "by_fact": fact_results,
        "results": test_results
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    print("\n" + "=" * 60)
    print(f"Вопросов: {metrics['total_questions']}")
    print(f"Hit@1: {metrics['hit_at_1'] * 100:.1f}%")
    print(f"Hit@{TOP_K}: {metrics[f'hit_at_{TOP_K}'] * 100:.1f}%")
    print(f"MRR: {metrics['mrr']:.3f}")
    print(f"Ошибок выполнения: {sum(item['status'] == 'ERROR' for item in test_results)}")
    print(f"Отчёт сохранён: {REPORT_FILE}")


if __name__ == "__main__":
    run_evaluation()
