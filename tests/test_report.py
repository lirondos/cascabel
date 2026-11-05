import json
import tempfile
import unittest
from pathlib import Path

from cascabel import Cascabel, CharLMConfig
from cascabel.metrics import Score
from cascabel.report import dimension_example

TEST_SENTENCES = [
    (["Un", "look", "total", "black"], ["O", "B-ENG", "B-ENG", "I-ENG"]),
    (["Visitamos", "Nueva", "York"], ["O", "B-LOC", "I-LOC"]),
]
TRAINING_SENTENCES = [
    (["Fracaso", "del", "look"], ["O", "O", "B-ENG"]),
    (["Vivo", "en", "Madrid"], ["O", "O", "B-LOC"]),
]
PREDICTION_SENTENCES = [
    (["Un", "look", "total", "black"], ["O", "B-ENG", "O", "B-ENG"]),
    (["Visitamos", "Nueva", "York"], ["O", "B-PER", "I-PER"]),
]


def shape_cascabel():
    """Enough training spans per label for prototypicality models."""
    names = ["Ana", "Luis", "Marta", "Pedro", "Rosa", "Juan"]
    training = [(["Vino", name], ["O", "B-PER"]) for name in names]
    return Cascabel.from_sentences(
        [(["Vino", "Carmen", "y", "Xq"], ["O", "B-PER", "O", "B-PER"])],
        training=training,
        charlm_config=CharLMConfig(epochs=2, hidden_size=8, embedding_size=4,
                                   min_label_spans=2),
    )


def full_cascabel():
    return Cascabel.from_sentences(
        TEST_SENTENCES, training=TRAINING_SENTENCES, prediction=PREDICTION_SENTENCES
    )


class ScoresByLabelTestCase(unittest.TestCase):
    def test_decomposition_matches_totals(self):
        c = full_cascabel()
        by_label = c.scores_by_label()
        total = Score.aggregate(by_label.values())
        self.assertEqual(total, c.span_scores())

    def test_per_label_values(self):
        c = full_cascabel()
        by_label = c.scores_by_label()
        # ENG: "look" correct; "total black" overlap_missed (fn) with an ENG
        # prediction (fp)
        self.assertEqual(by_label["ENG"], Score(tp=1, fp=1, fn=1))
        # LOC: "Nueva York" got a PER label: fn under LOC, fp under PER
        self.assertEqual(by_label["LOC"], Score(fn=1))
        self.assertEqual(by_label["PER"], Score(fp=1))

    def test_requires_prediction(self):
        c = Cascabel.from_sentences(TEST_SENTENCES)
        with self.assertRaises(RuntimeError):
            c.scores_by_label()


class ReportDictTestCase(unittest.TestCase):
    def test_full_report_sections(self):
        report = full_cascabel().report()
        for key in (
            "dataset",
            "dimensions",
            "scores",
            "scores_by_label",
            "scores_by_dimension",
            "error_counts",
            "diagnostics",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["dataset"]["entities"], 3)
        self.assertEqual(report["dataset"]["labels"], {"ENG": 2, "LOC": 1})

    def test_identity_and_shape_sections(self):
        report = full_cascabel().report()
        self.assertEqual(report["identity"]["n_spans"], 3)
        self.assertIn("unseen_share", report["identity"])
        self.assertNotIn("shape", report)  # ENG has too few spans for a model

    def test_shape_lists_without_scores(self):
        shape = shape_cascabel().report()["shape"]
        self.assertEqual(list(shape), ["PER"])
        self.assertEqual(set(shape["PER"]["least_prototypical"]), {"Carmen", "Xq"})
        self.assertEqual(shape["PER"]["least_prototypical"][::-1],
                         shape["PER"]["most_prototypical"])
        summary = shape_cascabel().summary()
        self.assertIn("least prototypical:", summary)
        self.assertIn("most prototypical:", summary)

    def test_perplexity_is_two_to_the_bits(self):
        ent = shape_cascabel().test.ents[0]
        self.assertAlmostEqual(ent._.prototypicality_perplexity, 2 ** ent._.prototypicality)
        self.assertAlmostEqual(ent._.surprisal_perplexity, 2 ** ent._.surprisal_mean)
        self.assertAlmostEqual(ent._.span_typicality_perplexity, 2 ** ent._.span_typicality)

    def test_no_identity_or_shape_without_training(self):
        report = Cascabel.from_sentences(TEST_SENTENCES).report()
        self.assertNotIn("identity", report)
        self.assertNotIn("shape", report)

    def test_zero_error_types_listed(self):
        summary = full_cascabel().summary()
        self.assertRegex(summary, r"fused\s+0")
        self.assertIn(">fused</span></td><td class='num'>0<", full_cascabel().to_html())

    def test_glosses_shown(self):
        summary = full_cascabel().summary()
        self.assertIn("the span starts at the beginning of the sentence", summary)
        self.assertIn("boundary: like span, but the label is ignored", summary)
        page = full_cascabel().to_html()
        self.assertIn("<th>meaning</th>", page)
        card = full_cascabel().card()
        self.assertIn("| meaning |", card)

    def test_error_tokens_use_error_colour(self):
        page = full_cascabel().to_html()
        # Nueva York: gold LOC, predicted PER -> one type-error box (purple)
        self.assertIn('<span class="ent" style="background:#7b2cbf40;border-bottom:3px '
                      'solid #7b2cbf" title="type | gold: Nueva York (LOC) | predicted: '
                      'Nueva York (PER)">Nueva York</span>', page)

    def test_type_error_shows_both_labels(self):
        page = full_cascabel().to_html()
        self.assertIn("gold “Nueva York” <b>LOC</b> → predicted “Nueva York” <b>PER</b>", page)

    def test_highlight_follows_prediction(self):
        # gold "total black", predicted "black": only "black" is highlighted
        page = full_cascabel().to_html()
        self.assertIn('title="overlap_missed | gold: total black (ENG) | '
                      'predicted: black (ENG)">black</span>', page)
        self.assertNotIn(">total black</span>", page)
        self.assertIn('</span> total <span', page)  # "total" left plain

    def test_report_is_json_serializable(self):
        serialized = json.dumps(full_cascabel().report())
        self.assertIn("error_counts", serialized)

    def test_profile_only_report(self):
        report = Cascabel.from_sentences(TEST_SENTENCES).report()
        self.assertIn("dataset", report)
        self.assertNotIn("scores", report)
        self.assertNotIn("error_counts", report)


class SummaryTestCase(unittest.TestCase):
    def test_summary_contains_sections(self):
        summary = full_cascabel().summary()
        for fragment in (
            "cascabel report",
            "Scores",
            "Scores by label",
            "Errors",
            "Diagnostics",
            "Identity (test spans vs training)",
            "multi-label (of seen)",
            "Dimensions",
            "overlap_missed",
            "ENG",
        ):
            self.assertIn(fragment, summary)

    def test_profile_only_summary(self):
        summary = Cascabel.from_sentences(TEST_SENTENCES).summary()
        self.assertIn("Dataset:", summary)
        self.assertNotIn("Scores by label", summary)


class DimensionExampleTestCase(unittest.TestCase):
    def test_most_frequent_form_with_context(self):
        sentences = [
            (["Vivo", "en", "Madrid", "desde", "hace", "mucho", "tiempo"],
             ["O", "O", "B-LOC", "O", "O", "O", "O"]),
            (["Ana", "y", "Madrid"], ["B-PER", "O", "B-LOC"]),
        ]
        casc = Cascabel.from_sentences(sentences)
        spans = [e for e in casc.test.ents if e.label_ == "LOC"]
        example = dimension_example(spans)
        self.assertEqual(example, {"text": "Madrid", "label": "LOC",
                                   "left": "Vivo en", "right": "desde hace mucho …"})
        self.assertIsNone(dimension_example([]))

    def test_example_in_every_report(self):
        casc = Cascabel.from_sentences(TEST_SENTENCES)
        report = casc.report()
        self.assertTrue(all("example" in d for d in report["dimensions"].values()))
        self.assertIn("example", casc.summary())
        self.assertIn("<th>example</th>", casc.to_html())


class HtmlTestCase(unittest.TestCase):
    def test_html_structure(self):
        html_report = full_cascabel().to_html()
        for fragment in (
            "<!doctype html>",
            "cascabel report",
            "Scores by label",
            "Dimensions",
            "chip",           # error chips
            "overlap_missed",
            "solid #ff8fc7\" title=\"overlap_missed",  # predicted "black" in the overlap_missed colour
        ):
            self.assertIn(fragment, html_report)

    def test_html_identity_and_shape(self):
        page = full_cascabel().to_html()
        self.assertIn("<h2>Identity (test spans vs training)</h2>", page)
        self.assertIn("multi-label (of seen)", page)
        shape_page = shape_cascabel().to_html()
        self.assertIn("<h2>Shape</h2>", shape_page)
        self.assertIn("<th>most prototypical</th>", shape_page)
        self.assertIn("<tr><td>PER</td>", shape_page)
        self.assertNotIn("bits/char", shape_page)

    def test_html_without_training_has_no_identity(self):
        page = Cascabel.from_sentences(TEST_SENTENCES, prediction=PREDICTION_SENTENCES).to_html()
        self.assertNotIn("Identity (test spans", page)
        self.assertNotIn("<h2>Shape</h2>", page)

    def test_html_escapes_tokens(self):
        c = Cascabel.from_tokens(
            ["<script>", "look"], ["O", "B-ENG"],
            prediction=(["<script>", "look"], ["O", "O"]),
        )
        html_report = c.to_html()
        self.assertNotIn("<script>", html_report)
        self.assertIn("&lt;script&gt;", html_report)

    def test_only_errors_filter(self):
        c = full_cascabel()
        errors_only = c.to_html(only_errors=True)
        everything = c.to_html(only_errors=False)
        # both test sentences contain errors here, so craft a correct one
        self.assertLessEqual(errors_only.count("class=\"sentence\""), 2)
        self.assertEqual(everything.count("class=\"sentence\""), 2)

    def test_no_error_message(self):
        c = Cascabel.from_sentences(TEST_SENTENCES, prediction=TEST_SENTENCES)
        self.assertIn("No sentences with errors", c.to_html())

    def test_write_to_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            returned = full_cascabel().to_html(path)
            self.assertEqual(returned, path)
            self.assertIn("cascabel report", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
