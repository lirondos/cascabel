import tempfile
import unittest
import warnings
from pathlib import Path

from spacy.tokens import Doc

from cascabel.doc import (
    bio_tag,
    doc_from_conll,
    doc_from_sentences,
    doc_from_tokens,
    load_wordlist,
    read_conll,
    repair_bio,
    repair_bio_sentences,
)


class BioTagTestCase(unittest.TestCase):
    def test_bio_tags(self):
        doc = doc_from_tokens(["Un", "look", "total"], ["O", "B-ENG", "I-ENG"])
        self.assertEqual([bio_tag(t) for t in doc], ["O", "B-ENG", "I-ENG"])


class RepairBioTestCase(unittest.TestCase):
    def test_valid_sequence_untouched(self):
        tags = ["O", "B-ENG", "I-ENG", "O", "B-LOC"]
        self.assertEqual(repair_bio(tags), tags)

    def test_i_after_o(self):
        self.assertEqual(repair_bio(["O", "I-ENG"]), ["O", "B-ENG"])

    def test_i_at_start(self):
        self.assertEqual(repair_bio(["I-ENG", "I-ENG"]), ["B-ENG", "I-ENG"])

    def test_i_after_different_label(self):
        self.assertEqual(repair_bio(["B-LOC", "I-ENG"]), ["B-LOC", "B-ENG"])

    def test_bare_i(self):
        self.assertEqual(repair_bio(["O", "I"]), ["O", "B"])

    def test_repair_sentences(self):
        repaired = repair_bio_sentences([(["a", "b"], ["O", "I-X"])])
        self.assertEqual(repaired, [(["a", "b"], ["O", "B-X"])])


class DocConstructionTestCase(unittest.TestCase):
    def test_from_tokens(self):
        doc = doc_from_tokens(["Un", "look", "total", "black"],
                              ["O", "B-ENG", "B-ENG", "I-ENG"])
        self.assertIsInstance(doc, Doc)
        self.assertEqual(len(doc), 4)
        entities = [(e.text, e.label_) for e in doc.ents]
        self.assertEqual(entities, [("look", "ENG"), ("total black", "ENG")])

    def test_from_sentences_boundaries(self):
        doc = doc_from_sentences([
            (["Hola", "mundo"], ["O", "O"]),
            (["Un", "look"], ["O", "B-ENG"]),
        ])
        self.assertEqual(len(list(doc.sents)), 2)
        self.assertEqual([t.is_sent_start for t in doc], [True, False, True, False])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            doc_from_tokens(["a", "b"], ["O"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            doc_from_sentences([])

    def test_ill_formed_bio_is_repaired_with_warning(self):
        with self.assertWarnsRegex(UserWarning, r"1 sentence.*sentence numbers: 2"):
            doc = doc_from_sentences([
                (["a", "b"], ["O", "B-ENG"]),
                (["c", "d"], ["O", "I-ENG"]),
            ])
        self.assertEqual([bio_tag(t) for t in doc], ["O", "B-ENG", "O", "B-ENG"])

    def test_well_formed_bio_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            doc_from_tokens(["a", "b"], ["B-ENG", "I-ENG"])

    def test_iob1_scheme_converts_without_warning(self):
        tags = ["I-PER", "I-PER", "B-PER", "O", "I-LOC"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            doc = doc_from_tokens(list("abcde"), tags, scheme="IOB1")
        self.assertEqual([bio_tag(t) for t in doc],
                         ["B-PER", "I-PER", "B-PER", "O", "B-LOC"])

    def test_bioes_scheme(self):
        tags = ["B-PER", "E-PER", "O", "S-LOC", "U-ORG", "B-X", "I-X", "L-X"]
        doc = doc_from_tokens(list("abcdefgh"), tags, scheme="bilou")
        self.assertEqual([bio_tag(t) for t in doc],
                         ["B-PER", "I-PER", "O", "B-LOC", "B-ORG",
                          "B-X", "I-X", "I-X"])

    def test_unknown_scheme_raises(self):
        with self.assertRaises(ValueError):
            doc_from_tokens(["a"], ["O"], scheme="XYZ")

    def test_strict_mode_raises_on_ill_formed_bio(self):
        with self.assertRaises(ValueError):
            doc_from_tokens(["a", "b"], ["O", "I-ENG"], repair=False)


class ConllTestCase(unittest.TestCase):
    def _write(self, content):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".conll", delete=False, encoding="utf-8"
        )
        handle.write(content)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_read_two_sentences(self):
        path = self._write("Un\tO\nlook\tB-ENG\n\ntotal\tB-ENG\nblack\tI-ENG\n")
        sentences = read_conll(path)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], (["Un", "look"], ["O", "B-ENG"]))
        self.assertEqual(sentences[1], (["total", "black"], ["B-ENG", "I-ENG"]))

    def test_no_trailing_newline(self):
        path = self._write("Un\tO\nlook\tB-ENG")
        sentences = read_conll(path)
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0][0], ["Un", "look"])

    def test_extra_columns_uses_last(self):
        path = self._write("Un X Y O\nlook X Y B-ENG\n")
        sentences = read_conll(path)
        self.assertEqual(sentences[0], (["Un", "look"], ["O", "B-ENG"]))

    def test_comments_skipped(self):
        path = self._write("# doc 1\nUn\tO\n")
        sentences = read_conll(path, comment_prefix="#")
        self.assertEqual(sentences[0][0], ["Un"])

    def test_docstart_skipped(self):
        path = self._write("-DOCSTART- -X- O\n\nUn O\nlook B-ENG\n-DOCSTART- -X- O\nwow B-ENG\n")
        sentences = read_conll(path)
        self.assertEqual(sentences, [(["Un", "look"], ["O", "B-ENG"]),
                                     (["wow"], ["B-ENG"])])
        kept = read_conll(path, skip_docstart=False)
        self.assertEqual(kept[0], (["-DOCSTART-"], ["O"]))

    def test_doc_from_conll(self):
        path = self._write("Un\tO\nlook\tB-ENG\n\nwow\tB-ENG\n")
        doc = doc_from_conll(path)
        self.assertEqual(len(list(doc.sents)), 2)
        self.assertEqual([e.text for e in doc.ents], ["look", "wow"])


class WordlistTestCase(unittest.TestCase):
    def test_load(self):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        )
        handle.write("Casa\n\n perro \nGATO\n")
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        self.assertEqual(load_wordlist(handle.name), {"casa", "perro", "gato"})
        self.assertEqual(
            load_wordlist(handle.name, lowercase=False), {"Casa", "perro", "GATO"}
        )


if __name__ == "__main__":
    unittest.main()
