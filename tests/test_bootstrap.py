import unittest

from cascabel import Cascabel

TEST = [
    (["Un", "look", "total", "black"], ["O", "B-ENG", "B-ENG", "I-ENG"]),
    (["Me", "gusta", "el", "shopping"], ["O", "O", "O", "B-ENG"]),
    (["El", "streaming", "es", "genial"], ["O", "B-ENG", "O", "O"]),
]
TRAIN = [(["Fracaso", "del", "look"], ["O", "O", "B-ENG"])]
PRED = [
    (["Un", "look", "total", "black"], ["O", "B-ENG", "O", "B-ENG"]),
    (["Me", "gusta", "el", "shopping"], ["O", "O", "O", "B-ENG"]),
    (["El", "streaming", "es", "genial"], ["O", "B-ENG", "O", "O"]),
]


class BootstrapTestCase(unittest.TestCase):
    def setUp(self):
        self.casc = Cascabel.from_sentences(TEST, training=TRAIN, prediction=PRED)

    def test_intervals_bracket_point(self):
        ci = self.casc.bootstrap(n=200, seed=1)
        for key in ("span", "token", "boundary"):
            self.assertIn(key, ci)
            entry = ci[key]
            self.assertLessEqual(entry["lo"], entry["point"] + 1e-9)
            self.assertGreaterEqual(entry["hi"], entry["point"] - 1e-9)
            self.assertLessEqual(entry["lo"], entry["hi"])

    def test_point_matches_scores(self):
        ci = self.casc.bootstrap(n=100, seed=2)
        self.assertAlmostEqual(ci["span"]["point"], self.casc.span_scores().f1)

    def test_reproducible_with_seed(self):
        a = self.casc.bootstrap(n=100, seed=7)
        b = self.casc.bootstrap(n=100, seed=7)
        self.assertEqual(a["span"], b["span"])

    def test_dimension_intervals_present(self):
        ci = self.casc.bootstrap(n=100, seed=3)
        # length bins are always available; f1 and recall both reported
        self.assertIn("length_2.f1", ci)
        self.assertIn("length_2.recall", ci)

    def test_requires_prediction(self):
        casc = Cascabel.from_sentences(TEST)
        with self.assertRaises(RuntimeError):
            casc.bootstrap(n=10)


if __name__ == "__main__":
    unittest.main()
