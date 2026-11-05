"""Tests for the error taxonomy: one scenario per error type."""

import unittest

from cascabel import Cascabel, ErrorType
from cascabel.metrics import Score


def analyze(tokens, gold, pred):
    """Single-sentence Cascabel with a prediction attached."""
    return Cascabel.from_tokens(tokens, gold, prediction=(tokens, pred))


class ExactMatchTestCase(unittest.TestCase):
    def test_correct(self):
        c = analyze(["Un", "look"], ["O", "B-ENG"], ["O", "B-ENG"])
        self.assertEqual(len(c.alignments), 1)
        alignment = c.alignments[0]
        self.assertFalse(alignment.is_error)
        self.assertIsNone(alignment.error_type)
        self.assertEqual(alignment.span_score, Score(tp=1))
        self.assertEqual(c.span_scores(), Score(tp=1))

    def test_correct_multitoken_boundary_score(self):
        c = analyze(["total", "black"], ["B-ENG", "I-ENG"], ["B-ENG", "I-ENG"])
        # one span with exact boundaries
        self.assertEqual(c.alignments[0].boundary_score, Score(tp=1))


class TypeSwapTestCase(unittest.TestCase):
    def test_type(self):
        c = analyze(["Paris"], ["B-LOC"], ["B-PER"])
        alignment = c.alignments[0]
        self.assertEqual(alignment.error_type, ErrorType.TYPE)
        self.assertEqual(alignment.span_score, Score(fn=1, fp=1))
        # boundaries are exact, so the label does not matter
        self.assertEqual(alignment.boundary_score, Score(tp=1))
        self.assertEqual(c.boundary_scores(), Score(tp=1))


class MissedSpuriousTestCase(unittest.TestCase):
    def test_missed(self):
        c = analyze(["Un", "look"], ["O", "B-ENG"], ["O", "O"])
        alignment = c.alignments[0]
        self.assertEqual(alignment.error_type, ErrorType.MISSED)
        self.assertEqual(alignment.span_score, Score(fn=1))
        self.assertEqual(alignment.boundary_score, Score(fn=1))

    def test_missed_multitoken_boundary(self):
        c = analyze(["total", "black"], ["B-ENG", "I-ENG"], ["O", "O"])
        # one gold span, not found
        self.assertEqual(c.alignments[0].boundary_score, Score(fn=1))

    def test_spurious(self):
        c = analyze(["Un", "look"], ["O", "O"], ["O", "B-ENG"])
        alignment = c.alignments[0]
        self.assertEqual(alignment.error_type, ErrorType.SPURIOUS)
        self.assertEqual(alignment.span_score, Score(fp=1))
        self.assertEqual(alignment.boundary_score, Score(fp=1))

    def test_disjoint_gold_and_pred(self):
        c = analyze(["a", "b", "c"], ["B-E", "O", "O"], ["O", "O", "B-E"])
        types = [a.error_type for a in c.alignments]
        self.assertEqual(types, [ErrorType.MISSED, ErrorType.SPURIOUS])

    def test_leftover_golds_are_missed(self):
        c = analyze(
            ["a", "b", "c"], ["B-E", "B-E", "B-E"], ["B-E", "O", "O"]
        )
        types = [a.error_type for a in c.alignments]
        self.assertEqual(types, [None, ErrorType.MISSED, ErrorType.MISSED])


class BoundaryScoreTestCase(unittest.TestCase):
    def test_labels_ignored_boundaries_must_be_exact(self):
        # gold: [Juan Peña]PER [Madrid]LOC; predicted: [Juan Peña]ORG [Madrid ,]LOC
        c = analyze(
            ["Juan", "Peña", "en", "Madrid", ","],
            ["B-PER", "I-PER", "O", "B-LOC", "O"],
            ["B-ORG", "I-ORG", "O", "B-LOC", "I-LOC"],
        )
        self.assertEqual(c.span_scores(), Score(fn=2, fp=2))
        self.assertEqual(c.boundary_scores(), Score(tp=1, fn=1, fp=1))


class FusedTestCase(unittest.TestCase):
    def test_fused(self):
        # gold: [look] [total black]; predicted: [look total black]
        c = analyze(
            ["un", "look", "total", "black"],
            ["O", "B-ENG", "B-ENG", "I-ENG"],
            ["O", "B-ENG", "I-ENG", "I-ENG"],
        )
        alignment = c.alignments[0]
        self.assertEqual(alignment.error_type, ErrorType.FUSED)
        self.assertEqual(len(alignment.gold), 2)
        self.assertEqual(len(alignment.predicted), 1)
        self.assertEqual(alignment.span_score, Score(fn=2, fp=1))
        # wrong token: the gold B that was not predicted as B ("total")
        self.assertEqual([t.text for t in alignment.wrong_tokens], ["total"])
        # no predicted span has the boundaries of a gold span
        self.assertEqual(alignment.boundary_score, Score(fn=2, fp=1))
        self.assertTrue(alignment.is_segmentation_error)


class SplitTestCase(unittest.TestCase):
    def test_split(self):
        # gold: [total black]; predicted: [total] [black]
        c = analyze(
            ["un", "look", "total", "black"],
            ["O", "O", "B-ENG", "I-ENG"],
            ["O", "O", "B-ENG", "B-ENG"],
        )
        alignment = c.alignments[0]
        self.assertEqual(alignment.error_type, ErrorType.SPLIT)
        self.assertEqual(alignment.span_score, Score(fn=1, fp=2))
        # wrong token: the predicted B that should have been I ("black")
        self.assertEqual([t.text for t in alignment.wrong_tokens], ["black"])
        self.assertEqual(alignment.boundary_score, Score(fn=1, fp=2))
        self.assertTrue(alignment.is_segmentation_error)


class OverlapTestCase(unittest.TestCase):
    def test_overlap_missed(self):
        # predicted entity strictly inside the gold entity
        c = analyze(
            ["new", "york", "city"],
            ["B-LOC", "I-LOC", "I-LOC"],
            ["O", "B-LOC", "O"],
        )
        alignment = c.alignments[0]
        self.assertEqual(alignment.error_type, ErrorType.OVERLAP_MISSED)
        self.assertEqual(alignment.span_score, Score(fn=1, fp=1))

    def test_overlap_spurious(self):
        # predicted entity strictly containing the gold entity
        c = analyze(
            ["new", "york", "city"],
            ["O", "B-LOC", "O"],
            ["B-LOC", "I-LOC", "I-LOC"],
        )
        alignment = c.alignments[0]
        self.assertEqual(alignment.error_type, ErrorType.OVERLAP_SPURIOUS)
        self.assertEqual(alignment.span_score, Score(fn=1, fp=1))

    def test_overlap_spurious_missed(self):
        # staggered: prediction shifted right of the gold entity
        c = analyze(
            ["a", "b", "c", "d"],
            ["B-E", "I-E", "O", "O"],
            ["O", "B-E", "I-E", "O"],
        )
        alignment = c.alignments[0]
        self.assertEqual(alignment.error_type, ErrorType.OVERLAP_SPURIOUS_MISSED)


class MissegmentedTestCase(unittest.TestCase):
    def test_boundary_drift(self):
        # gold: [leggings animal] [print]; predicted: [leggings] [animal print]
        c = analyze(
            ["leggings", "animal", "print"],
            ["B-ENG", "I-ENG", "B-ENG"],
            ["B-ENG", "B-ENG", "I-ENG"],
        )
        self.assertEqual(len(c.alignments), 1)
        alignment = c.alignments[0]
        self.assertEqual(alignment.error_type, ErrorType.MISSEGMENTED)
        self.assertEqual(len(alignment.gold), 2)
        self.assertEqual(len(alignment.predicted), 2)
        self.assertEqual(alignment.span_score, Score(fn=2, fp=2))

    def test_missed_plus_missegmented(self):
        # gold: [a b] [c d]; predicted: [b c d]
        c = analyze(
            ["a", "b", "c", "d"],
            ["B-E", "I-E", "B-E", "I-E"],
            ["O", "B-E", "I-E", "I-E"],
        )
        types = [a.error_type for a in c.alignments]
        self.assertEqual(types, [ErrorType.MISSED, ErrorType.MISSEGMENTED])
        self.assertEqual(c.alignments[0].gold[0].text, "a b")
        self.assertEqual(c.alignments[1].gold[0].text, "c d")
        self.assertEqual(c.alignments[1].predicted[0].text, "b c d")

    def test_straddling_single(self):
        # gold: [a b] [c d]; predicted: [a b c] — straddles into the second
        c = analyze(
            ["a", "b", "c", "d"],
            ["B-E", "I-E", "B-E", "I-E"],
            ["B-E", "I-E", "I-E", "O"],
        )
        types = [a.error_type for a in c.alignments]
        self.assertEqual(types, [ErrorType.MISSEGMENTED, ErrorType.MISSED])


    def test_extra_token_before_next_entity(self):
        # gold: [SIALKOT] , [Pakistan]; predicted: [SIALKOT ,] [Pakistan]
        c = analyze(
            ["SIALKOT", ",", "Pakistan", "1996"],
            ["B-LOC", "O", "B-LOC", "O"],
            ["B-LOC", "I-LOC", "B-LOC", "O"],
        )
        types = [a.error_type for a in c.alignments]
        self.assertEqual(types, [ErrorType.OVERLAP_SPURIOUS, None])
        self.assertEqual(c.alignments[0].predicted[0].text, "SIALKOT ,")
        self.assertEqual(c.alignments[1].predicted[0].text, "Pakistan")

    def test_perfect_straddle_keeps_both_predictions(self):
        # gold: [a b] [c d]; predicted: [a b c] [d]
        c = analyze(
            ["a", "b", "c", "d"],
            ["B-E", "I-E", "B-E", "I-E"],
            ["B-E", "I-E", "I-E", "B-E"],
        )
        self.assertEqual(len(c.alignments), 1)
        alignment = c.alignments[0]
        self.assertEqual(alignment.error_type, ErrorType.MISSEGMENTED)
        self.assertEqual([p.text for p in alignment.predicted], ["a b c", "d"])
        self.assertEqual(alignment.span_score, Score(fn=2, fp=2))


class MultiSentenceTestCase(unittest.TestCase):
    def test_alignment_is_per_sentence(self):
        c = Cascabel.from_sentences(
            [(["Un", "look"], ["O", "B-ENG"]), (["Total", "black"], ["B-ENG", "I-ENG"])],
            prediction=[(["Un", "look"], ["O", "B-ENG"]), (["Total", "black"], ["O", "O"])],
        )
        self.assertEqual(len(c.alignments), 2)
        self.assertIsNone(c.alignments[0].error_type)
        self.assertEqual(c.alignments[1].error_type, ErrorType.MISSED)
        self.assertEqual(c.span_scores(), Score(tp=1, fn=1))


class ErrorSummariesTestCase(unittest.TestCase):
    def test_error_counts(self):
        c = analyze(
            ["a", "b", "c", "d"],
            ["B-E", "O", "B-E", "O"],
            ["O", "O", "B-E", "B-E"],
        )
        counts = c.error_counts()
        self.assertEqual(counts["missed"], 1)
        self.assertEqual(counts["spurious"], 1)
        self.assertEqual(len(c.errors()), 2)


if __name__ == "__main__":
    unittest.main()
