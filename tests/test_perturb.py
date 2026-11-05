import unittest

from cascabel import add_quotes, recase, register_extensions, strip_quotes
from cascabel.doc import bio_tag, doc_from_sentences, doc_from_tokens


class RecaseTestCase(unittest.TestCase):
    def setUp(self):
        register_extensions()
        self.doc = doc_from_tokens(
            ["Un", "Look", "TOTAL", "black"], ["O", "B-ENG", "B-ENG", "I-ENG"]
        )

    def test_lowercase_doc(self):
        lowered = recase(self.doc, "lower")
        self.assertEqual([t.text for t in lowered], ["un", "look", "total", "black"])
        self.assertEqual([bio_tag(t) for t in lowered],
                         ["O", "B-ENG", "B-ENG", "I-ENG"])

    def test_uppercase_doc(self):
        uppered = recase(self.doc, "upper")
        self.assertEqual([t.text for t in uppered], ["UN", "LOOK", "TOTAL", "BLACK"])

    def test_titlecase_doc(self):
        titled = recase(self.doc, "title")
        self.assertEqual([t.text for t in titled], ["Un", "Look", "Total", "Black"])

    def test_entities_scope(self):
        lowered = recase(self.doc, "lower", scope="entities")
        self.assertEqual([t.text for t in lowered], ["Un", "look", "total", "black"])

    def test_sentence_boundaries_preserved(self):
        doc = doc_from_sentences([(["Hola"], ["O"]), (["Adiós"], ["O"])])
        lowered = recase(doc, "lower")
        self.assertEqual(len(list(lowered.sents)), 2)

    def test_invalid_case(self):
        with self.assertRaises(ValueError):
            recase(self.doc, "camel")

    def test_invalid_scope(self):
        with self.assertRaises(ValueError):
            recase(self.doc, "lower", scope="everything")


class AddQuotesTestCase(unittest.TestCase):
    def setUp(self):
        register_extensions()

    def test_wraps_unquoted_entities(self):
        doc = doc_from_tokens(["Un", "look", "total"], ["O", "B-ENG", "O"])
        quoted = add_quotes(doc)
        self.assertEqual([t.text for t in quoted],
                         ["Un", '"', "look", '"', "total"])
        self.assertEqual([bio_tag(t) for t in quoted],
                         ["O", "O", "B-ENG", "O", "O"])
        # the freshly quoted entity is now detected as quoted
        self.assertTrue(quoted.ents[0]._.is_quotated)

    def test_already_quoted_untouched(self):
        doc = doc_from_tokens(["Un", '"', "look", '"', "total"],
                              ["O", "O", "B-ENG", "O", "O"])
        quoted = add_quotes(doc)
        self.assertEqual(len(quoted), len(doc))

    def test_multitoken_entity(self):
        doc = doc_from_tokens(["Un", "total", "black"], ["O", "B-ENG", "I-ENG"])
        quoted = add_quotes(doc)
        self.assertEqual([t.text for t in quoted],
                         ["Un", '"', "total", "black", '"'])
        self.assertEqual([e.text for e in quoted.ents], ["total black"])

    def test_with_casing(self):
        doc = doc_from_tokens(["Un", "LOOK", "total"], ["O", "B-ENG", "O"])
        quoted = add_quotes(doc, doc_case="lower")
        self.assertEqual([t.text for t in quoted],
                         ["un", '"', "look", '"', "total"])

    def test_entity_case_only(self):
        doc = doc_from_tokens(["Un", "LOOK", "total"], ["O", "B-ENG", "O"])
        quoted = add_quotes(doc, entity_case="lower")
        self.assertEqual([t.text for t in quoted],
                         ["Un", '"', "look", '"', "total"])


class StripQuotesTestCase(unittest.TestCase):
    def setUp(self):
        register_extensions()

    def test_strips_quoted_entities(self):
        doc = doc_from_tokens(["Un", '"', "look", '"', "total"],
                              ["O", "O", "B-ENG", "O", "O"])
        stripped = strip_quotes(doc)
        self.assertEqual([t.text for t in stripped], ["Un", "look", "total"])
        self.assertEqual([e.text for e in stripped.ents], ["look"])

    def test_unquoted_untouched(self):
        doc = doc_from_tokens(["Un", "look", "total"], ["O", "B-ENG", "O"])
        stripped = strip_quotes(doc)
        self.assertEqual(len(stripped), len(doc))

    def test_roundtrip(self):
        doc = doc_from_tokens(["Un", "look", "total"], ["O", "B-ENG", "O"])
        roundtrip = strip_quotes(add_quotes(doc))
        self.assertEqual([t.text for t in roundtrip], [t.text for t in doc])
        self.assertEqual([bio_tag(t) for t in roundtrip], [bio_tag(t) for t in doc])

    def test_sentence_initial_quote_inheritance(self):
        doc = doc_from_tokens(['"', "Look", '"', "total"],
                              ["O", "B-ENG", "O", "O"])
        # the opening quote held the sentence-initial position; after
        # stripping, the entity's first token must inherit it
        stripped = strip_quotes(doc)
        self.assertEqual([t.text for t in stripped], ["Look", "total"])
        self.assertTrue(stripped[0].is_sent_start)
        self.assertEqual([e.text for e in stripped.ents], ["Look"])


if __name__ == "__main__":
    unittest.main()
