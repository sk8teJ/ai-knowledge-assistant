import unittest

from evaluation.run_evaluation import check_answer, normalize_text
from evaluation.evaluate_retrieval import calculate_metrics


class AnswerEvaluationTests(unittest.TestCase):
    def test_normalize_text_ignores_case_and_extra_whitespace(self):
        self.assertEqual(normalize_text("  ТЕСТ   текста "), "тест текста")

    def test_answerable_question_accepts_expected_phrase(self):
        self.assertTrue(check_answer(
            "Компания была основана в 2018 году.",
            ["2018"],
            True
        ))

    def test_answerable_question_rejects_missing_expected_phrase(self):
        self.assertFalse(check_answer(
            "Компания была основана в 2019 году.",
            ["2018"],
            True
        ))

    def test_answerable_question_without_expected_answers_fails_safely(self):
        self.assertFalse(check_answer("Любой ответ", [], True))

    def test_unanswerable_question_requires_refusal_marker(self):
        self.assertTrue(check_answer(
            "В предоставленных документах нет этой информации.",
            [],
            False
        ))

    def test_unanswerable_question_rejects_fabricated_answer(self):
        self.assertFalse(check_answer(
            "Генеральный директор — Иван Иванов.",
            [],
            False
        ))


class RetrievalMetricTests(unittest.TestCase):
    def test_metrics_for_mixed_ranks(self):
        results = [
            {"relevant_rank": 1},
            {"relevant_rank": 2},
            {"relevant_rank": None},
            {"relevant_rank": 3}
        ]
        metrics = calculate_metrics(results, top_k=3)
        self.assertEqual(metrics["total_questions"], 4)
        self.assertEqual(metrics["hit_at_1"], 0.25)
        self.assertEqual(metrics["hit_at_3"], 0.75)
        self.assertEqual(metrics["mrr"], round((1 + 0.5 + 0 + (1 / 3)) / 4, 4))

    def test_empty_metric_dataset(self):
        metrics = calculate_metrics([], top_k=3)
        self.assertEqual(metrics["total_questions"], 0)
        self.assertEqual(metrics["hit_at_3"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)


if __name__ == "__main__":
    unittest.main()
