import math
import unittest

from cascabel.charlm import CharLM, CharLMConfig, ShapeModels, span_text

SPANISH_TOKENS = [
    "casa", "perro", "gato", "comida", "mesa", "silla", "ventana", "puerta",
    "camino", "camisa", "cabeza", "corazón", "tiempo", "trabajo", "palabra",
    "momento", "manera", "persona", "mundo", "vida", "parte", "lugar",
]

CONFIG = CharLMConfig(epochs=40, hidden_size=32, embedding_size=16, seed=0)


class CharLMTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = CharLM(CONFIG).fit(SPANISH_TOKENS)

    def test_surprisal_ordering(self):
        # English-shaped tokens are more surprising under a Spanish model.
        shopping, casa = self.model.bits_per_char(["shopping", "casa"])
        self.assertGreater(shopping, casa)

    def test_bits_per_char_matches_total(self):
        total = self.model.log2_probabilities(["casa"])[0]
        self.assertAlmostEqual(self.model.bits_per_char(["casa"])[0], -total / 5)

    def test_unknown_characters(self):
        score = self.model.bits_per_char(["中文"])[0]
        self.assertTrue(math.isfinite(score))

    def test_empty_string_scores_end_event(self):
        self.assertTrue(math.isfinite(self.model.bits_per_char([""])[0]))

    def test_deterministic_with_seed(self):
        other = CharLM(CONFIG).fit(SPANISH_TOKENS)
        self.assertEqual(
            other.log2_probabilities(["perro", "shopping"]),
            self.model.log2_probabilities(["perro", "shopping"]),
        )

    def test_case_preserved(self):
        lower, upper = self.model.bits_per_char(["casa", "CASA"])
        self.assertNotAlmostEqual(lower, upper)

    def test_empty_fit_raises(self):
        with self.assertRaises(ValueError):
            CharLM(CONFIG).fit(["", ""])

    def test_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            CharLM(CONFIG).bits_per_char(["casa"])


class ShapeModelsTestCase(unittest.TestCase):
    def test_label_models_and_min_spans(self):
        spans = [("PER", "Juan Pérez")] * 3 + [("LOC", "Madrid")]
        config = CharLMConfig(epochs=2, hidden_size=8, embedding_size=4,
                              min_label_spans=2)
        models = ShapeModels(SPANISH_TOKENS, spans, config=config)
        self.assertEqual(list(models.label_models), ["PER"])
        typicality, fit, other = models.prototypicality([("PER", "Ana")])[0]
        self.assertIsNotNone(typicality)
        self.assertIsNone(fit)  # no other label to compare with
        self.assertIsNone(other)
        self.assertEqual(models.prototypicality([("LOC", "Roma")])[0][:2],
                         (None, None))
        # the any-span model covers every label, rare ones included
        self.assertIsNotNone(models.span_typicality(["Roma"])[0])

    def test_span_typicality_without_spans(self):
        models = ShapeModels(SPANISH_TOKENS, [], config=CONFIG)
        self.assertIsNone(models.span_model)
        self.assertEqual(models.span_typicality(["Madrid", "Ana"]), [None, None])

    def test_span_typicality_ordering(self):
        spans = [("PER", "Juan Pérez"), ("PER", "Ana García"),
                 ("LOC", "Madrid"), ("LOC", "Toledo")] * 5
        models = ShapeModels(SPANISH_TOKENS, spans, config=CONFIG)
        familiar, odd = models.span_typicality(["Ana Pérez", "xqzw kkj"])
        self.assertLess(familiar, odd)

    def test_empty_background_raises(self):
        with self.assertRaises(ValueError):
            ShapeModels([], config=CONFIG)

    def test_cascabel_annotates_test_spans(self):
        from cascabel import Cascabel
        training = [(["Juan", "Pérez", "vive", "en", "Madrid"],
                     ["B-PER", "I-PER", "O", "O", "B-LOC"])] * 3
        test = [(["Ana", "García", "come", "en", "Toledo"],
                  ["B-PER", "I-PER", "O", "O", "B-LOC"])]
        config = CharLMConfig(epochs=2, hidden_size=8, embedding_size=4,
                              min_label_spans=2)
        casc = Cascabel.from_sentences(test, training=training,
                                       charlm_config=config)
        rows = casc.span_table("test")
        for row in rows:
            self.assertIsNotNone(row["span_typicality"])
            self.assertIsNotNone(row["prototypicality"])
        # training spans are what the models learned from: never scored
        for row in casc.span_table("training"):
            self.assertIsNone(row["span_typicality"])
            self.assertIsNone(row["prototypicality"])
            self.assertIsNone(row["surprisal_mean"])

    def test_extreme_spans(self):
        from cascabel import Cascabel
        training = [(["Juan", "Pérez", "vive", "en", "Madrid"],
                     ["B-PER", "I-PER", "O", "O", "B-LOC"])] * 3
        test = [(["Ana", "García", "y", "Juan", "Pérez", "en", "Madrid"],
                  ["B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-LOC"]),
                (["Madrid"], ["B-LOC"])]
        config = CharLMConfig(epochs=2, hidden_size=8, embedding_size=4,
                              min_label_spans=2)
        casc = Cascabel.from_sentences(test, training=training,
                                       charlm_config=config)
        top = casc.extreme_spans("span_typicality", n=2)
        self.assertEqual(len(top), 2)
        self.assertGreaterEqual(top[0]["span_typicality"], top[1]["span_typicality"])
        bottom = casc.extreme_spans("span_typicality", n=3, least=True)
        self.assertEqual(bottom, sorted(bottom, key=lambda r: r["span_typicality"]))
        per = casc.extreme_spans("length", label="PER")
        self.assertEqual({r["label"] for r in per}, {"PER"})
        texts = [r["text"] for r in casc.extreme_spans("length", n=10)]
        self.assertEqual(len(texts), len(set(texts)))  # distinct spans
        unseen = casc.extreme_spans("length", unseen_only=True)
        self.assertEqual([r["text"] for r in unseen], ["Ana García"])
        madrid = [r for r in casc.extreme_spans("length", n=10) if r["text"] == "Madrid"]
        self.assertEqual(madrid[0]["occurrences"], 2)
        with self.assertRaises(ValueError):
            casc.extreme_spans("casing")

    def test_warns_when_test_is_training(self):
        from cascabel import Cascabel
        sentences = [(["Juan", "vive", "en", "Madrid"],
                      ["B-PER", "O", "O", "B-LOC"])]
        with self.assertWarnsRegex(UserWarning, "same tokens as the training"):
            Cascabel.from_sentences(sentences, training=sentences, shape=False)

    def test_span_text(self):
        self.assertEqual(span_text(["Bank", "of", "England"]), "Bank of England")


if __name__ == "__main__":
    unittest.main()
