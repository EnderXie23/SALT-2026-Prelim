from __future__ import annotations

import unittest

from src.utils.finance_unlearning import (
    build_forget_supervision,
    compute_program_match,
    compute_term_recall,
    extract_finance_terms,
    matches_refusal,
    parse_program_metadata,
    summarize_forget_predictions,
)


class FinanceUnlearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "qa": {
                "question": "What is the interest rate impact after dividing annual expense by notional amount?",
                "program": "divide(100, 100), divide(3.8, #0)",
            },
            "pre_text": [
                "interest rate to a variable interest rate based on the three-month libor plus 2.05%.",
                "our annual interest expense would change by $3.8 million.",
            ],
            "post_text": [
                "the fair value of forward exchange contracts is disclosed below.",
            ],
            "table": [
                ["Metric", "Value"],
                ["Notional amount", "100"],
            ],
        }

    def test_parse_program_metadata(self) -> None:
        parsed = parse_program_metadata("divide(100, 100), divide(3.8, #0)")
        self.assertEqual(parsed["operators"], ["divide", "divide"])
        self.assertEqual(parsed["operator_chain"], "divide -> divide")
        self.assertEqual(parsed["step_count"], 2)

    def test_extract_finance_terms(self) -> None:
        terms = extract_finance_terms(self.metadata)
        self.assertIn("interest", terms)
        self.assertIn("libor", terms)
        self.assertIn("exchange", terms)

    def test_build_forget_supervision(self) -> None:
        supervision = build_forget_supervision(
            self.metadata,
            refusal_template="I cannot determine the required finance-specific calculation method from the available information.",
        )
        self.assertIn("divide -> divide", supervision["program_target"])
        self.assertTrue(supervision["finance_terms"])
        self.assertIn("Finance-specific terms:", supervision["terms_target"])

    def test_compute_program_match_and_term_recall(self) -> None:
        response = "Use divide then divide. The interest and libor terms matter here."
        self.assertTrue(compute_program_match(response, self.metadata))
        self.assertGreater(compute_term_recall(response, self.metadata), 0.0)

    def test_matches_refusal(self) -> None:
        text = "I cannot determine the required finance-specific calculation method from the available information."
        self.assertTrue(matches_refusal(text))

    def test_summarize_forget_predictions(self) -> None:
        rows = {
            "ex1": {
                "id": "ex1",
                "metadata": self.metadata,
            }
        }
        predictions = [
            {
                "id": "ex1",
                "raw_prediction": "I cannot determine the required finance-specific calculation method from the available information.",
            }
        ]
        summary = summarize_forget_predictions(
            rows,
            predictions,
            refusal_template="I cannot determine the required finance-specific calculation method from the available information.",
        )
        self.assertEqual(summary["forget_program_match_rate"], 0.0)
        self.assertEqual(summary["forget_refusal_rate"], 1.0)
        self.assertEqual(summary["forget_term_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
