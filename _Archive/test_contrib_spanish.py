import unittest

try:
    import pylabeador  # noqa: F401

    HAS_PYLABEADOR = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_PYLABEADOR = False

from cascabel import Cascabel, dimension_report


@unittest.skipUnless(HAS_PYLABEADOR, "pylabeador not installed")
class SpanishPhonotacticsTestCase(unittest.TestCase):
    def setUp(self):
        from cascabel.contrib import spanish

        self.spanish = spanish

    def test_anomalous_words(self):
        for word in ("shopping", "crush", "look", "streaming", "flash"):
            self.assertTrue(
                self.spanish.is_phonotactically_anomalous(word), word
            )

    def test_compliant_words(self):
        for word in ("casa", "perro", "azul", "trabajo", "canción", "grande"):
            self.assertFalse(
                self.spanish.is_phonotactically_anomalous(word), word
            )

    def test_non_alphabetic(self):
        self.assertFalse(self.spanish.is_phonotactically_anomalous("123"))
        self.assertFalse(self.spanish.is_phonotactically_anomalous(""))

    def test_backwards_compatible_alias(self):
        self.assertIs(
            self.spanish.is_perplexing, self.spanish.is_phonotactically_anomalous
        )

    def test_as_custom_dimension(self):
        c = Cascabel.from_tokens(
            ["Un", "look", "de", "moda"],
            ["O", "B-ENG", "B-ENG", "I-ENG"],
        )
        dims = {"phonotactic_anomaly": lambda ent: any(
            self.spanish.is_phonotactically_anomalous(t.text)
            for t in ent if t.text.isalpha())}
        report = dimension_report(c.test, dims)
        self.assertEqual([e.text for e in report["phonotactic_anomaly"].spans],
                         ["look"])


if __name__ == "__main__":
    unittest.main()
