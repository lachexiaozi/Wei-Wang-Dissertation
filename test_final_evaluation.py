import unittest

from Final_Evaluation import (
    _classification_metrics,
    optimal_one_to_one_match,
)


def prediction(bearing):
    return {"bearing_deg": bearing}


def target(identifier, bearing):
    return {"ground_truth_id": identifier, "bearing_deg": bearing}


class FinalEvaluationTests(unittest.TestCase):
    def test_assignment_maximizes_cardinality_before_error(self):
        predictions = [prediction(0.6), prediction(1.4)]
        ground_truth = [target("GT1", 0.0), target("GT2", 1.0)]
        self.assertEqual(
            optimal_one_to_one_match(predictions, ground_truth, 1.0),
            [0, 1],
        )

    def test_each_ground_truth_is_used_at_most_once(self):
        predictions = [prediction(0.0), prediction(0.1)]
        ground_truth = [target("GT1", 0.0)]
        assignment = optimal_one_to_one_match(predictions, ground_truth, 1.0)
        self.assertEqual(sum(item is not None for item in assignment), 1)
        self.assertEqual(assignment[0], 0)

    def test_collinear_ground_truth_retains_one_false_negative(self):
        predictions = [prediction(0.0)]
        ground_truth = [target("GT1", 0.0), target("GT2", 0.0)]
        assignment = optimal_one_to_one_match(predictions, ground_truth, 1.0)
        self.assertEqual(assignment, [0])
        self.assertEqual(len(ground_truth) - 1, 1)

    def test_empty_predictions_match_nothing(self):
        self.assertEqual(
            optimal_one_to_one_match([], [target("GT1", 0.0)], 1.0),
            [],
        )

    def test_metrics(self):
        precision, recall, f1 = _classification_metrics(4, 1, 1)
        self.assertAlmostEqual(precision, 0.8)
        self.assertAlmostEqual(recall, 0.8)
        self.assertAlmostEqual(f1, 0.8)


if __name__ == "__main__":
    unittest.main()
