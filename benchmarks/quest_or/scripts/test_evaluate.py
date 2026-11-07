import unittest

from evaluate import evaluate, set_metrics


class QuestScoringTests(unittest.TestCase):
    def test_worked_example(self):
        result = set_metrics(['A', 'B', 'C'], ['A', 'B', 'D', 'E'])
        self.assertEqual(result['precision'], 0.5)
        self.assertEqual(result['recall'], 2 / 3)
        self.assertEqual(result['f1'], 4 / 7)

    def test_duplicates_are_not_extra_answers(self):
        self.assertEqual(set_metrics(['A', 'B'], ['A', 'A', 'B'])['f1'], 1)

    def test_empty_and_incorrect_answers(self):
        self.assertEqual(set_metrics(['A'], [])['f1'], 0)
        self.assertEqual(set_metrics(['A'], ['B'])['f1'], 0)

    def test_macro_average_gives_queries_equal_weight(self):
        gold = [
            {'query': 'one', 'docs': ['A'], 'metadata': {'domain': 'x', 'template': 'or'}},
            {'query': 'two', 'docs': list('BCDEFGHIJK'), 'metadata': {'domain': 'x', 'template': 'or'}},
        ]
        pred = [{'query': 'one', 'docs': ['A']}, {'query': 'two', 'docs': []}]
        summary, _ = evaluate(gold, pred)
        self.assertEqual(summary['overall']['f1'], 0.5)

    def test_missing_predictions_are_rejected(self):
        gold = [{'query': 'one', 'docs': ['A'], 'metadata': {'domain': 'x', 'template': 'or'}}]
        with self.assertRaises(ValueError):
            evaluate(gold, [])


if __name__ == '__main__':
    unittest.main()
