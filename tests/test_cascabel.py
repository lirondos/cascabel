"""Integration tests for the Cascabel facade."""

import tempfile
import unittest
from pathlib import Path

from cascabel import Cascabel, ErrorType
from cascabel.metrics import Score

TEST_SENTENCES = [
    (["Un", "look", "total", "black"], ["O", "B-ENG", "B-ENG", "I-ENG"]),
    (["Me", "gusta", "el", "shopping"], ["O", "O", "O", "B-ENG"]),
]
TRAINING_SENTENCES = [
    (["Fracaso", "total", "del", "look"], ["O", "O", "O", "B-ENG"]),
    (["El", "streaming", "es", "popular"], ["O", "B-ENG", "O", "O"]),
]
PREDICTION_SENTENCES = [
    (["Un", "look", "total", "black"], ["O", "B-ENG", "O", "B-ENG"]),
    (["Me", "gusta", "el", "shopping"], ["O", "O", "O", "B-ENG"]),
]


class FacadeTestCase(unittest.TestCase):
    def setUp(self):
        self.cascabel = Cascabel.from_sentences(
            TEST_SENTENCES,
            training=TRAINING_SENTENCES,
            prediction=PREDICTION_SENTENCES,
        )

    def test_span_scores(self):
        # sentence 1: "look" correct, "total black" vs "black" overlap_missed
        # sentence 2: "shopping" correct
        self.assertEqual(self.cascabel.span_scores(), Score(tp=2, fp=1, fn=1))

    def test_token_scores(self):
        # gold non-O: look B, total B, black I, shopping B
        # predictions: look B ✓, total O (fn), black B (≠ I: fn + fp), shopping ✓
        self.assertEqual(self.cascabel.token_scores(), Score(tp=2, fp=1, fn=2))

    def test_error_counts(self):
        self.assertEqual(dict(self.cascabel.error_counts()),
                         {"overlap_missed": 1})

    def test_retrieval_capacity(self):
        # 3 of 4 gold entity tokens were labeled entity ("total" was not)
        self.assertAlmostEqual(self.cascabel.retrieval_capacity(), 3 / 4)

    def test_noise_level(self):
        # no gold-O token was labeled entity
        self.assertAlmostEqual(self.cascabel.noise_level(), 0.0)

    def test_segmentation_rates(self):
        self.assertAlmostEqual(self.cascabel.segmentation_error_rate(), 0.0)
        # "total black" is adjacent to "look" and was not matched exactly
        self.assertAlmostEqual(self.cascabel.segmentation_success_rate(), 0.0)

    def test_scores_by_dimension(self):
        by_dimension = self.cascabel.scores_by_dimension()
        # the single two-token gold entity ("total black") was in error
        self.assertEqual(by_dimension["length_2"], Score(fn=1, fp=1))
        # both single-token entities were correct
        self.assertEqual(by_dimension["length_1"], Score(tp=2))

    def test_span_outcomes_two_sides(self):
        gold = [(span.text, outcome)
                for span, outcome in self.cascabel.span_outcomes("gold")]
        self.assertIn(("total black", "overlap_missed"), gold)
        self.assertIn(("look", None), gold)
        predicted = [(span.text, outcome)
                     for span, outcome in self.cascabel.span_outcomes("predicted")]
        # the boundary error appears on both sides, each with its own span
        self.assertIn(("black", "overlap_missed"), predicted)
        with self.assertRaises(ValueError):
            self.cascabel.span_outcomes("both")

    def test_precision_by_dimension(self):
        by_dimension = self.cascabel.scores_by_dimension(side="predicted")
        # predicted spans: "look" (ok), "black" (error), "shopping" (ok);
        # all three are one token long
        self.assertEqual(by_dimension["length_1"], Score(tp=2, fp=1))
        self.assertNotIn("length_2", by_dimension)

    def test_error_profile(self):
        gold = self.cascabel.error_profile("gold")
        row = gold["length_2"]
        self.assertEqual((row["n"], row["correct"], row["errors"]), (1, 0, 1))
        self.assertEqual(row["overlap_missed"], 1)
        self.assertAlmostEqual(row["share_of_correct"], 0.0)
        self.assertAlmostEqual(row["share_of_errors"], 100.0)
        self.assertAlmostEqual(row["rate"], 0.0)
        self.assertAlmostEqual(gold["length_1"]["share_of_correct"], 100.0)
        predicted = self.cascabel.error_profile("predicted")
        self.assertAlmostEqual(predicted["length_1"]["rate"], 2 / 3)

    def test_span_table(self):
        rows = self.cascabel.span_table("test")
        self.assertEqual(len(rows), 3)
        by_text = {row["text"]: row for row in rows}
        self.assertEqual(by_text["total black"]["outcome"], "overlap_missed")
        self.assertEqual(by_text["look"]["outcome"], "correct")
        self.assertEqual(by_text["total black"]["length"], 2)
        self.assertEqual(by_text["total black"]["casing"], "lower")
        predicted = {row["text"]: row for row in self.cascabel.span_table("prediction")}
        self.assertEqual(predicted["black"]["outcome"], "overlap_missed")

    def test_descriptive_statistics(self):
        self.assertEqual(self.cascabel.n_entities(), 3)
        self.assertEqual(self.cascabel.n_sentences(), 2)
        self.assertEqual(self.cascabel.n_tokens(), 8)
        self.assertEqual(self.cascabel.label_distribution(), {"ENG": 3})
        self.assertEqual(self.cascabel.span_length_distribution(), {1: 2, 2: 1})
        self.assertEqual(
            self.cascabel.spans_per_sentence_distribution(), {2: 1, 1: 1}
        )



class PredictionValidationTestCase(unittest.TestCase):
    def test_scores_require_prediction(self):
        c = Cascabel.from_sentences(TEST_SENTENCES)
        with self.assertRaises(RuntimeError):
            c.span_scores()

    def test_token_count_mismatch(self):
        c = Cascabel.from_sentences(TEST_SENTENCES)
        with self.assertRaises(ValueError):
            c.set_prediction_from_sentences([(["Un"], ["O"])])

    def test_text_mismatch(self):
        c = Cascabel.from_tokens(["Un", "look"], ["O", "B-ENG"])
        with self.assertRaises(ValueError):
            c.set_prediction_from_tokens(["Un", "outfit"], ["O", "B-ENG"])

    def test_sentence_boundary_mismatch(self):
        c = Cascabel.from_sentences([(["a", "b"], ["O", "O"])])
        with self.assertRaises(ValueError):
            c.set_prediction_from_sentences([(["a"], ["O"]), (["b"], ["O"])])

    def test_prediction_repair(self):
        c = Cascabel.from_tokens(["Un", "look"], ["O", "B-ENG"])
        with self.assertWarnsRegex(UserWarning, "ill-formed BIO"):
            c.set_prediction_from_tokens(["Un", "look"], ["O", "I-ENG"])
        self.assertEqual(c.test[1]._.iob_pred, "B-ENG")
        self.assertEqual(c.span_scores(), Score(tp=1))

    def test_replacing_prediction_recomputes(self):
        c = Cascabel.from_sentences(TEST_SENTENCES, prediction=TEST_SENTENCES)
        self.assertEqual(c.span_scores(), Score(tp=3))
        c.set_prediction_from_sentences(PREDICTION_SENTENCES)
        self.assertEqual(c.span_scores(), Score(tp=2, fp=1, fn=1))


class ConllConstructorTestCase(unittest.TestCase):
    def _write(self, sentences):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".conll", delete=False, encoding="utf-8"
        )
        for tokens, tags in sentences:
            for token, tag in zip(tokens, tags):
                handle.write(f"{token}\t{tag}\n")
            handle.write("\n")
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_from_conll(self):
        test_path = self._write(TEST_SENTENCES)
        training_path = self._write(TRAINING_SENTENCES)
        prediction_path = self._write(PREDICTION_SENTENCES)
        c = Cascabel.from_conll(
            test_path, training=training_path, prediction=prediction_path
        )
        self.assertEqual(c.span_scores(), Score(tp=2, fp=1, fn=1))
        self.assertIsNotNone(c.shape_models)


class PerfectPredictionTestCase(unittest.TestCase):
    def test_all_scores_perfect(self):
        c = Cascabel.from_sentences(TEST_SENTENCES, prediction=TEST_SENTENCES)
        for score in (c.span_scores(), c.token_scores(), c.boundary_scores()):
            self.assertEqual(score.f1, 1.0)
        self.assertEqual(c.errors(), [])
        self.assertEqual(c.retrieval_capacity(), 1.0)
        self.assertEqual(c.noise_level(), 0.0)


class NoEntityEdgeCaseTestCase(unittest.TestCase):
    def test_no_entities_anywhere(self):
        sentences = [(["Hola", "mundo"], ["O", "O"])]
        c = Cascabel.from_sentences(sentences, prediction=sentences)
        self.assertEqual(c.span_scores(), Score())
        self.assertEqual(c.retrieval_capacity(), 0.0)
        self.assertEqual(c.segmentation_error_rate(), 0.0)
        self.assertEqual(c.segmentation_success_rate(), 0.0)


if __name__ == "__main__":
    unittest.main()
