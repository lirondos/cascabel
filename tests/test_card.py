import tempfile
import unittest
from pathlib import Path

from cascabel import Cascabel

TEST = [(["Un", "look", "total", "black"], ["O", "B-ENG", "B-ENG", "I-ENG"])]
TRAIN = [(["Fracaso", "del", "look"], ["O", "O", "B-ENG"])]
PRED = [(["Un", "look", "total", "black"], ["O", "B-ENG", "O", "B-ENG"])]


class CardTestCase(unittest.TestCase):
    def test_output_card_sections(self):
        c = Cascabel.from_sentences(TEST, training=TRAIN, prediction=PRED)
        card = c.card(dataset="demo", system="model-x")
        for fragment in ("Performance Profile Card", "demo", "model-x",
                         "Composition", "Metrics",
                         "Error profile", "Disaggregated evaluation",
                         "Caveats and recommendations", "overlap_missed"):
            self.assertIn(fragment, card)

    def test_profile_card_without_prediction(self):
        card = Cascabel.from_sentences(TEST).card(dataset="demo")
        self.assertIn("Dataset Profile Card", card)
        self.assertNotIn("Metrics", card)
        # missing training is an auto-caveat
        self.assertIn("No training split", card)

    def test_small_test_caveat(self):
        card = Cascabel.from_sentences(TEST, training=TRAIN).card()
        self.assertIn("Small evaluation set", card)

    def test_identity_and_shape_sections(self):
        card = Cascabel.from_sentences(TEST, training=TRAIN, prediction=PRED).card()
        self.assertIn("## Identity (test spans vs training)", card)
        self.assertIn("| unseen label (of seen) |", card)

    def test_small_dimension_caveat(self):
        card = Cascabel.from_sentences(TEST, training=TRAIN, prediction=PRED).card()
        self.assertIn("Fewer than 30 gold spans in", card)
        self.assertIn("`length_1` (n=1)", card)

    def test_caseless_na_caveat(self):
        c = Cascabel.from_sentences(
            [(["苹果", "收购", "微软"], ["B-ORG", "O", "B-ORG"])])
        card = c.card(dataset="zh-demo")
        self.assertNotIn("| casing_title |", card)  # omitted entirely (all n/a)

    def test_write_to_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.md"
            returned = Cascabel.from_sentences(TEST).card(path, dataset="d")
            self.assertEqual(returned, path)
            self.assertIn("Dataset Profile Card",
                          path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
