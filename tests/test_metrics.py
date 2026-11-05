import unittest

from cascabel.metrics import Score


class ScoreTestCase(unittest.TestCase):
    def test_perfect(self):
        score = Score(tp=10)
        self.assertEqual(score.precision, 1.0)
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.f1, 1.0)

    def test_zero_guards(self):
        score = Score()
        self.assertEqual(score.precision, 0.0)
        self.assertEqual(score.recall, 0.0)
        self.assertEqual(score.f1, 0.0)

    def test_known_values(self):
        score = Score(tp=3, fp=1, fn=2)
        self.assertAlmostEqual(score.precision, 3 / 4)
        self.assertAlmostEqual(score.recall, 3 / 5)
        self.assertAlmostEqual(score.f1, 2 * (0.75 * 0.6) / (0.75 + 0.6))

    def test_addition(self):
        total = Score(tp=1, fp=2, fn=3) + Score(tp=4, fp=5, fn=6)
        self.assertEqual(total, Score(tp=5, fp=7, fn=9))

    def test_aggregate(self):
        total = Score.aggregate([Score(tp=1), Score(fp=1), Score(fn=1)])
        self.assertEqual(total, Score(tp=1, fp=1, fn=1))
        self.assertEqual(Score.aggregate([]), Score())

    def test_sum_builtin(self):
        total = sum([Score(tp=1), Score(tp=2)], Score())
        self.assertEqual(total.tp, 3)

    def test_as_dict(self):
        data = Score(tp=1, fp=1, fn=0).as_dict()
        self.assertEqual(data["tp"], 1)
        self.assertAlmostEqual(data["precision"], 0.5)
        self.assertAlmostEqual(data["recall"], 1.0)

    def test_immutability(self):
        score = Score(tp=1)
        with self.assertRaises(Exception):
            score.tp = 2


if __name__ == "__main__":
    unittest.main()
