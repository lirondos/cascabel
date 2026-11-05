import unittest

from cascabel import (
    Cascabel,
    CharLMConfig,
    dimension_report,
    fold_text,
    register_extensions,
)
from cascabel.doc import doc_from_sentences, doc_from_tokens


class SurfaceExtensionTestCase(unittest.TestCase):
    def setUp(self):
        register_extensions()

    def test_multitoken(self):
        doc = doc_from_tokens(["Un", "look", "total", "black"],
                              ["O", "B-ENG", "B-ENG", "I-ENG"])
        self.assertFalse(doc.ents[0]._.is_multitoken)
        self.assertTrue(doc.ents[1]._.is_multitoken)

    def test_sentence_initial(self):
        doc = doc_from_tokens(["Look", "total", "de", "Zara"],
                              ["B-ENG", "O", "O", "B-ORG"])
        self.assertTrue(doc.ents[0]._.is_sentence_initial)
        self.assertFalse(doc.ents[1]._.is_sentence_initial)

    def test_sentence_initial_after_punctuation(self):
        doc = doc_from_tokens(["¿", "Look", "total", "?"],
                              ["O", "B-ENG", "O", "O"])
        self.assertTrue(doc.ents[0]._.is_sentence_initial)

    def test_quoted(self):
        doc = doc_from_tokens(["Un", '"', "look", '"', "total"],
                              ["O", "O", "B-ENG", "O", "O"])
        self.assertTrue(doc.ents[0]._.is_quotated)

    def test_not_quoted(self):
        doc = doc_from_tokens(["Un", "look", "total"], ["O", "B-ENG", "O"])
        self.assertFalse(doc.ents[0]._.is_quotated)

    def test_casing_is_independent_of_position(self):
        doc = doc_from_tokens(["Zara", "compra", "Bershka"],
                              ["B-ORG", "O", "B-ORG"])
        self.assertEqual(doc.ents[0]._.casing, "title")  # sentence initial
        self.assertEqual(doc.ents[1]._.casing, "title")

    def test_casing_values(self):
        cases = {
            ("NASA",): "upper",
            ("NATO", "Summit"): "title",
            ("Bank", "of", "England"): "title",
            ("Banco", "de", "España"): "title",
            ("McDonald",): "title",
            ("New", "york"): "title",
            ("Ñúñez",): "title",
            ("3M",): "upper",
            ("'s", "Hertogenbosch"): "mixed",
            ("iphone",): "lower",
            ("gripe", "aviar"): "lower",
            ("iPhone",): "mixed",
            ("eBay", "Inc."): "mixed",
        }
        for tokens, expected in cases.items():
            tags = ["B-X"] + ["I-X"] * (len(tokens) - 1)
            doc = doc_from_tokens(["la"] + list(tokens), ["O"] + tags)
            self.assertEqual(doc.ents[0]._.casing, expected, tokens)

    def test_length(self):
        doc = doc_from_tokens(["Un", "look", "total", "black"],
                              ["O", "B-ENG", "B-ENG", "I-ENG"])
        self.assertEqual([e._.length for e in doc.ents], [1, 2])

    def test_adjacent(self):
        doc = doc_from_tokens(["Un", "look", "total", "black"],
                              ["O", "B-ENG", "B-ENG", "I-ENG"])
        self.assertFalse(doc.ents[0]._.is_adjacent)
        self.assertTrue(doc.ents[1]._.is_adjacent)


class CaselessScriptTestCase(unittest.TestCase):
    def setUp(self):
        register_extensions()

    def test_casing_not_applicable_for_caseless_text(self):
        doc = doc_from_tokens(["苹果", "收购", "公司"], ["B-ORG", "O", "O"])
        self.assertIsNone(doc.ents[0]._.casing)

    def test_cased_material_in_caseless_context(self):
        doc = doc_from_tokens(["我", "用", "iPhone", "手机"],
                              ["O", "O", "B-PROD", "I-PROD"])
        # "iPhone 手机" contains cased characters: casing applies
        self.assertEqual(doc.ents[0]._.casing, "mixed")

    def test_report_omits_fully_inapplicable_dimension(self):
        doc = doc_from_tokens(["苹果", "收购", "微软"],
                              ["B-ORG", "O", "B-ORG"])
        report = dimension_report(doc)
        self.assertNotIn("casing_title", report)
        self.assertNotIn("casing_upper", report)
        self.assertIn("length_1", report)

    def test_percentage_over_applicable_only(self):
        # one caseless entity, one titlecased (mid-sentence) entity
        doc = doc_from_tokens(["我", "用", "Google", "和", "苹果"],
                              ["O", "O", "B-ORG", "O", "B-ORG"])
        report = dimension_report(doc)
        titlecase = report["casing_title"]
        self.assertEqual(titlecase.total, 2)
        self.assertEqual(titlecase.applicable, 1)
        self.assertEqual(titlecase.count, 1)
        self.assertAlmostEqual(titlecase.percentage, 100.0)
        self.assertAlmostEqual(titlecase.applicability, 50.0)

    def test_digits_are_caseless(self):
        doc = doc_from_tokens(["call", "911", "now"], ["O", "B-NUM", "O"])
        self.assertIsNone(doc.ents[0]._.casing)


class DimensionReportTestCase(unittest.TestCase):
    def test_surface_only_report(self):
        doc = doc_from_tokens(["Un", "look", "total", "black"],
                              ["O", "B-ENG", "B-ENG", "I-ENG"])
        register_extensions()
        report = dimension_report(doc)
        self.assertEqual(report["length_2"].count, 1)
        self.assertEqual(report["length_1"].count, 1)
        self.assertEqual(report["length_3"].count, 0)
        self.assertEqual(report["adjacent"].count, 1)
        self.assertAlmostEqual(report["length_2"].percentage, 50.0)
        # training-dependent and lexicon dimensions unavailable: omitted
        self.assertNotIn("unseen", report)
        self.assertNotIn("in_lexicon", report)
        self.assertNotIn("partial_in_lexicon", report)

    def test_length_bins(self):
        tokens = ["a"] + ["X"] * 7
        tags = ["O", "B-X", "I-X", "I-X", "I-X", "B-X", "I-X", "I-X"]
        doc = doc_from_tokens(tokens, tags)
        register_extensions()
        report = dimension_report(doc)
        self.assertEqual(report["length_4_5"].count, 1)
        self.assertEqual(report["length_3"].count, 1)
        self.assertEqual(report["length_6plus"].count, 0)

    def test_lexicon_dimensions_absent_with_non_alphabetic_spans(self):
        doc = doc_from_tokens(["en", "1860", "y", "Roma"],
                              ["O", "B-DATE", "O", "B-LOC"])
        register_extensions()
        self.assertNotIn("in_lexicon", dimension_report(doc))

    def test_training_dimensions_present(self):
        c = Cascabel.from_tokens(
            ["Un", "look", "total", "black"],
            ["O", "B-ENG", "B-ENG", "I-ENG"],
            training=(["Fracaso", "del", "look"], ["O", "O", "B-ENG"]),
        )
        report = c.dimension_report()
        self.assertIn("unseen", report)
        # "look" was seen as an entity in training; "total black" was not
        self.assertEqual(report["unseen"].count, 1)
        self.assertEqual([e.text for e in report["unseen"].spans], ["total black"])


class TrainingAnnotationTestCase(unittest.TestCase):
    def setUp(self):
        self.cascabel = Cascabel.from_sentences(
            [(["Un", "look", "total", "black"], ["O", "B-ENG", "B-ENG", "I-ENG"])],
            training=[
                (["Fracaso", "total", "del", "look"], ["O", "O", "O", "B-ENG"]),
                (["El", "look", "es", "total"], ["O", "B-ENG", "O", "O"]),
            ],
        )

    def test_unseen_tokens(self):
        tokens = {t.text: t._.is_unseen_token for t in self.cascabel.test}
        self.assertFalse(tokens["look"])
        self.assertFalse(tokens["total"])
        self.assertTrue(tokens["black"])
        self.assertTrue(tokens["Un"])

    def test_unseen_tagging(self):
        # "total" was seen in training but always as O, never as B-ENG
        total = self.cascabel.test[2]
        self.assertEqual(total.text, "total")
        self.assertTrue(total._.is_unseen_tagging)
        # "look" was seen as B-ENG
        look = self.cascabel.test[1]
        self.assertFalse(look._.is_unseen_tagging)

    def test_entity_level_unseen(self):
        entities = {e.text: e._.is_unseen for e in self.cascabel.test.ents}
        self.assertFalse(entities["look"])
        self.assertTrue(entities["total black"])

    def test_entity_level_counts(self):
        entities = {e.text: e for e in self.cascabel.test.ents}
        self.assertEqual(entities["look"]._.train_count, 2)
        self.assertEqual(entities["look"]._.label_consistency, 1.0)
        self.assertEqual(entities["look"]._.identity, "seen_same_label")
        self.assertEqual(entities["total black"]._.train_count, 0)
        self.assertIsNone(entities["total black"]._.label_consistency)
        self.assertEqual(entities["total black"]._.identity, "unseen")


class IdentityTestCase(unittest.TestCase):
    """Identity compares the exact word sequence with every training
    occurrence and its BIO tags, ``O`` and ``I-`` included."""

    def identity_of(self, test, training):
        c = Cascabel.from_sentences([test], training=training)
        return c.test.ents[0]

    def test_ambiguous_label(self):
        ent = self.identity_of(
            (["visit", "Washington", "hoy"], ["O", "B-LOC", "O"]),
            [(["Washington", "dijo"], ["B-PER", "O"]),
             (["en", "Washington"], ["O", "B-LOC"])],
        )
        self.assertEqual(ent._.train_count, 2)
        self.assertAlmostEqual(ent._.label_consistency, 0.5)
        self.assertEqual(ent._.identity, "seen_ambiguous")
        self.assertFalse(ent._.is_unseen)

    def test_seen_only_as_o_is_other_label(self):
        ent = self.identity_of(
            (["Apple", "sube"], ["B-ORG", "O"]),
            [(["una", "Apple", "roja"], ["O", "O", "O"])],
        )
        self.assertEqual(ent._.train_count, 1)
        self.assertEqual(ent._.label_consistency, 0.0)
        self.assertEqual(ent._.identity, "seen_other_label")

    def test_inside_longer_span_does_not_count(self):
        # rule B: Madrid inside Comunidad de Madrid / Real Madrid is not
        # an occurrence of the span Madrid
        ent = self.identity_of(
            (["Madrid", "crece"], ["B-LOC", "O"]),
            [(["en", "Madrid"], ["O", "B-LOC"]),
             (["Comunidad", "de", "Madrid"], ["B-LOC", "I-LOC", "I-LOC"]),
             (["el", "Real", "Madrid"], ["O", "B-ORG", "I-ORG"])],
        )
        self.assertEqual(ent._.identity, "seen_same_label")
        self.assertEqual(ent._.train_labels, {"LOC": 1})

    def test_span_start_of_longer_span_does_not_count(self):
        # Madrid as B-LOC at the start of Madrid Barajas is not a whole span
        ent = self.identity_of(
            (["Madrid", "crece"], ["B-LOC", "O"]),
            [(["Madrid", "Barajas"], ["B-LOC", "I-LOC"])],
        )
        self.assertTrue(ent._.is_unseen)

    def test_seen_only_inside_longer_span_is_unseen(self):
        ent = self.identity_of(
            (["Yastrzhembsky", "dijo"], ["B-PER", "O"]),
            [(["Sergei", "Yastrzhembsky"], ["B-PER", "I-PER"])],
        )
        self.assertTrue(ent._.is_unseen)
        self.assertEqual(ent._.word_exposure, {"in_span": 1, "only_O": 0, "unseen": 0})
        self.assertTrue(ent._.has_word_seen_in_span)

    def test_span_match_any_keeps_old_behaviour(self):
        c = Cascabel.from_sentences(
            [(["Madrid", "crece"], ["B-LOC", "O"])],
            training=[(["en", "Madrid"], ["O", "B-LOC"]),
                      (["Comunidad", "de", "Madrid"], ["B-LOC", "I-LOC", "I-LOC"]),
                      (["el", "Real", "Madrid"], ["O", "B-ORG", "I-ORG"])],
            span_match="any", shape=False,
        )
        ent = c.test.ents[0]
        self.assertEqual(ent._.train_labels, {"LOC": 2, "ORG": 1})
        self.assertEqual(ent._.identity, "seen_ambiguous")

    def test_bad_span_match(self):
        with self.assertRaises(ValueError):
            Cascabel.from_sentences([(["a"], ["O"])], span_match="tokens")

    def test_matching_is_case_sensitive(self):
        ent = self.identity_of(
            (["TOKYO", "hoy"], ["B-LOC", "O"]),
            [(["en", "Tokyo"], ["O", "B-LOC"])],
        )
        self.assertEqual(ent._.train_count, 0)
        self.assertTrue(ent._.is_unseen)

    def test_multitoken_needs_same_tag_sequence(self):
        ent = self.identity_of(
            (["New", "York", "hoy"], ["B-LOC", "I-LOC", "O"]),
            [(["New", "York", "Times"], ["B-ORG", "I-ORG", "I-ORG"]),
             (["en", "New", "York"], ["O", "B-LOC", "I-LOC"]),
             (["en", "New", "York"], ["O", "B-LOC", "I-LOC"]),
             (["el", "New", "York"], ["O", "B-ORG", "I-ORG"])],
        )
        self.assertEqual(ent._.train_count, 3)  # New York Times does not count
        self.assertAlmostEqual(ent._.label_consistency, 2 / 3)
        self.assertAlmostEqual(ent._.majority_share, 2 / 3)
        self.assertEqual(ent._.majority_label, "LOC")
        self.assertEqual(ent._.train_labels, {"LOC": 2, "ORG": 1})

    def test_partly_o_occurrence_does_not_count(self):
        ent = self.identity_of(
            (["de", "Madrid"], ["B-LOC", "I-LOC"]),
            [(["vive", "de", "Madrid"], ["O", "O", "B-LOC"])],
        )
        self.assertTrue(ent._.is_unseen)

    def test_majority_can_be_another_tagging(self):
        ent = self.identity_of(
            (["Apple", "sube"], ["B-ORG", "O"]),
            [(["una", "Apple", "roja"], ["O", "O", "O"]),
             (["otra", "Apple"], ["O", "O"]),
             (["Apple", "gana"], ["B-ORG", "O"])],
        )
        self.assertAlmostEqual(ent._.label_consistency, 1 / 3)
        self.assertAlmostEqual(ent._.majority_share, 2 / 3)
        self.assertEqual(ent._.majority_label, "O")

    def test_unseen_has_no_majority(self):
        ent = self.identity_of(
            (["Zara", "abre"], ["B-ORG", "O"]),
            [(["Mango", "abre"], ["B-ORG", "O"])],
        )
        self.assertIsNone(ent._.majority_share)
        self.assertIsNone(ent._.majority_label)
        self.assertEqual(ent._.train_labels, {})

    def test_identity_distribution(self):
        c = Cascabel.from_sentences(
            [(["Madrid", "y", "Zara"], ["B-LOC", "O", "B-ORG"])],
            training=[(["en", "Madrid"], ["O", "B-LOC"])],
        )
        self.assertEqual(c.identity_distribution("test"),
                         {"seen_same_label": 1, "unseen": 1})

    def test_token_layer(self):
        c = Cascabel.from_sentences(
            [(["Bank", "of", "Tokyo-Mitsubishi"], ["B-ORG", "I-ORG", "I-ORG"])],
            training=[(["Bank", "of", "England"], ["B-ORG", "I-ORG", "I-ORG"]),
                      (["a", "cup", "of", "tea"], ["O", "O", "O", "O"])],
            shape=False,
        )
        bank, of, tokyo = c.test
        self.assertEqual(bank._.train_labels, {"ORG": 1})
        self.assertEqual(of._.train_labels, {"ORG": 1, "O": 1})
        self.assertEqual(tokyo._.train_labels, {})
        self.assertEqual([t._.exposure for t in c.test], ["in_span", "in_span", "unseen"])
        ent = c.test.ents[0]
        self.assertEqual(ent._.word_exposure, {"in_span": 2, "only_O": 0, "unseen": 1})

    def test_word_seen_only_as_o_is_not_familiar(self):
        ent = self.identity_of(
            (["Black", "Rhino"], ["B-ORG", "I-ORG"]),
            [(["the", "Black", "market"], ["O", "O", "O"])],
        )
        self.assertEqual(ent._.word_exposure, {"in_span": 0, "only_O": 1, "unseen": 1})
        self.assertFalse(ent._.has_word_seen_in_span)

    def test_identity_summary(self):
        c = Cascabel.from_sentences(
            [(["Madrid", "y", "Zara", "y", "Yastrzhembsky", "y", "Apple"],
              ["B-LOC", "O", "B-ORG", "O", "B-PER", "O", "B-ORG"])],
            training=[(["en", "Madrid"], ["O", "B-LOC"]),
                      (["Sergei", "Yastrzhembsky"], ["B-PER", "I-PER"]),
                      (["una", "Apple"], ["O", "O"])],
            shape=False,
        )
        summary = c.identity_summary("test")
        self.assertEqual(summary["n_spans"], 4)
        self.assertEqual(summary["n_unseen"], 2)  # Zara, Yastrzhembsky
        self.assertAlmostEqual(summary["unseen_share"], 0.5)
        self.assertEqual(summary["n_unseen_familiar_word"], 1)  # Yastrzhembsky
        self.assertAlmostEqual(summary["unseen_familiar_word_share"], 0.5)
        self.assertEqual(summary["n_seen"], 2)
        self.assertEqual(summary["n_ambiguous"], 1)  # Apple (seen as O)
        self.assertAlmostEqual(summary["ambiguous_share"], 0.5)
        self.assertEqual(summary["n_multi_label"], 0)
        self.assertEqual(summary["n_unseen_label"], 1)  # Apple (only O)

    def test_radar_axes(self):
        c = Cascabel.from_sentences(
            [(["Madrid", "y", "Lyon", "y", "Bonn", "y", "Roma", "y", "Zara"],
              ["B-LOC", "O", "B-PER", "O", "B-LOC", "O", "B-LOC", "O", "B-ORG"])],
            training=[(["en", "Madrid"], ["O", "B-LOC"]),
                      (["en", "Lyon"], ["O", "B-LOC"]),
                      (["el", "Lyon"], ["O", "B-ORG"]),
                      (["el", "Bonn"], ["O", "B-ORG"]),
                      (["en", "Roma"], ["O", "B-LOC"]),
                      (["Roma", "dijo"], ["O", "O"])],
            shape=False,
        )
        summary = c.identity_summary("test")
        self.assertAlmostEqual(summary["unseen_share"], 1 / 5)  # Zara
        self.assertEqual(summary["n_seen"], 4)
        self.assertEqual(summary["n_multi_label"], 2)  # Lyon (LOC, ORG), Roma (LOC, O)
        self.assertAlmostEqual(summary["multi_label_share"], 0.5)
        self.assertEqual(summary["n_unseen_label"], 2)  # Lyon (PER), Bonn (LOC)
        self.assertAlmostEqual(summary["unseen_label_share"], 0.5)

    def test_identity_summary_without_training(self):
        c = Cascabel.from_sentences([(["Madrid"], ["B-LOC"])], shape=False)
        self.assertEqual(c.identity_summary("test"), {})

    def test_madrid_walkthrough(self):
        c = Cascabel.from_sentences(
            [(["en", "Madrid"], ["O", "B-LOC"])],
            training=[(["en", "Madrid"], ["O", "B-LOC"]),
                      (["el", "Madrid"], ["O", "B-ORG"]),
                      (["Comunidad", "de", "Madrid"], ["B-ORG", "I-ORG", "I-ORG"]),
                      (["Madrid", "dijo"], ["O", "O"]),
                      (["en", "MADRID"], ["O", "B-LOC"])],
            shape=False,
        )
        ent = c.test.ents[0]
        self.assertEqual(ent._.train_labels, {"LOC": 1, "ORG": 1, "O": 1})
        self.assertEqual(ent._.identity, "seen_ambiguous")
        self.assertEqual(ent._.variant_labels, {"LOC": 1})  # MADRID
        madrid = c.test[1]
        self.assertEqual(madrid._.train_labels, {"ORG": 2, "LOC": 1, "O": 1})
        self.assertEqual(madrid._.variant_labels, {"LOC": 1})

    def test_el_pais_walkthrough(self):
        c = Cascabel.from_sentences(
            [(["El", "País", "dijo"], ["B-ORG", "I-ORG", "O"])],
            training=[(["El", "País", "dijo"], ["B-ORG", "I-ORG", "O"]),
                      (["El", "País", "Vasco"], ["B-LOC", "I-LOC", "I-LOC"]),
                      (["el", "país", "crece"], ["O", "O", "O"]),
                      (["El", "País", "dijo"], ["O", "B-ORG", "O"]),
                      (["El", "País", "dijo"], ["O", "O", "O"]),
                      (["EL", "PAIS", "dijo"], ["B-ORG", "I-ORG", "O"])],
            shape=False,
        )
        ent = c.test.ents[0]
        self.assertEqual(ent._.train_labels, {"ORG": 1, "O": 1})
        self.assertEqual(ent._.identity, "seen_ambiguous")
        # el país (O) and EL PAIS (ORG); the partial El [País] and the
        # longer El País Vasco do not count, as for exact matches.
        self.assertEqual(ent._.variant_labels, {"O": 1, "ORG": 1})
        el, pais = c.test[0], c.test[1]
        self.assertEqual(el._.train_labels, {"O": 2, "ORG": 1, "LOC": 1})
        self.assertEqual(el._.variant_labels, {"O": 1, "ORG": 1})  # el, EL
        self.assertEqual(pais._.variant_labels, {"O": 1, "ORG": 1})  # país, PAIS

    def test_unseen_span_seen_as_variant(self):
        c = Cascabel.from_sentences(
            [(["Japan", "y", "Zara"], ["B-LOC", "O", "B-ORG"])],
            training=[(["SOCCER", "-", "JAPAN", "WIN"], ["O", "O", "B-LOC", "O"])],
            shape=False,
        )
        japan, zara = c.test.ents
        self.assertEqual(japan._.identity, "unseen")
        self.assertEqual(japan._.variant_labels, {"LOC": 1})
        self.assertEqual(zara._.variant_labels, {})
        summary = c.identity_summary("test")
        self.assertEqual(summary["n_unseen_variant"], 1)
        self.assertAlmostEqual(summary["unseen_variant_share"], 0.5)

    def test_fold_text(self):
        self.assertEqual(fold_text("EL PAÍS"), "el pais")
        self.assertEqual(fold_text("Málaga"), fold_text("MALAGA"))
        self.assertEqual(fold_text("PEÑA"), "peña")  # ñ is kept
        self.assertNotEqual(fold_text("año"), fold_text("ano"))
        self.assertEqual(fold_text("Ñúñez"), "ñuñez")


    def test_matches_do_not_cross_sentences(self):
        ent = self.identity_of(
            (["New", "York", "hoy"], ["B-LOC", "I-LOC", "O"]),
            [(["algo", "New"], ["O", "O"]), (["York", "dijo"], ["B-PER", "O"])],
        )
        self.assertTrue(ent._.is_unseen)

    def test_identity_dimensions(self):
        c = Cascabel.from_sentences(
            [(["Apple", "y", "Washington", "y", "look", "y", "Zed"],
              ["B-ORG", "O", "B-LOC", "O", "B-ENG", "O", "B-PER"])],
            training=[
                (["una", "Apple"], ["O", "O"]),
                (["Washington", "dijo"], ["B-PER", "O"]),
                (["en", "Washington"], ["O", "B-LOC"]),
                (["el", "look"], ["O", "B-ENG"]),
            ],
        )
        report = c.dimension_report()
        spans = {name: [e.text for e in report[name].spans]
                 for name in ("unseen", "seen_same_label",
                              "seen_ambiguous", "seen_other_label")}
        self.assertEqual(spans, {"unseen": ["Zed"], "seen_same_label": ["look"],
                                 "seen_ambiguous": ["Washington"],
                                 "seen_other_label": ["Apple"]})
        self.assertNotIn("unseen_tagging", report)
        self.assertNotIn("type_confusable", report)

    def test_without_training_identity_is_none(self):
        doc = doc_from_tokens(["Un", "look"], ["O", "B-ENG"])
        register_extensions()
        self.assertIsNone(doc.ents[0]._.identity)
        self.assertIsNone(doc.ents[0]._.is_unseen)


class LabelAmbiguityTestCase(unittest.TestCase):
    def test_multi_label_entities_in_training(self):
        training = [
            (["en", "Madrid"], ["O", "B-LOC"]),
            (["en", "Madrid"], ["O", "B-LOC"]),
            (["el", "Real", "Madrid"], ["O", "B-ORG", "I-ORG"]),
            (["Ana", "vive"], ["B-PER", "O"]),
            (["Apple", "sube"], ["B-ORG", "O"]),
            (["una", "Apple"], ["O", "O"]),
        ]
        c = Cascabel.from_sentences(training[:1], training=training, shape=False)
        result = c.label_ambiguity()
        self.assertEqual(result["n_entities"], 4)  # Madrid, Real Madrid, Ana, Apple
        self.assertEqual(result["n_multi_label"], 1)  # Apple (Madrid in Real Madrid not counted)
        self.assertAlmostEqual(result["multi_label_share"], 0.25)
        by_text = {e["text"]: e for e in result["entities"]}
        self.assertEqual(by_text["Madrid"]["labels"], {"LOC": 2})
        self.assertAlmostEqual(by_text["Madrid"]["majority_share"], 1.0)
        self.assertEqual(by_text["Apple"]["labels"], {"ORG": 1, "O": 1})
        self.assertEqual(result["entities"][0]["text"], "Apple")  # most ambiguous first

    def test_occurrence_label(self):
        from cascabel.doc import occurrence_label
        self.assertEqual(occurrence_label(["O", "O"]), "O")
        self.assertEqual(occurrence_label(["I-LOC"]), "LOC")
        self.assertEqual(occurrence_label(["B-ORG", "I-ORG"]), "ORG")
        self.assertEqual(occurrence_label(["O", "B-LOC"]), "MIXED")
        self.assertEqual(occurrence_label(["I-ORG", "B-LOC"]), "MIXED")

class WordlistAnnotationTestCase(unittest.TestCase):
    def test_lexicon_ambiguity(self):
        c = Cascabel.from_tokens(
            ["Un", "look", "total", "black"],
            ["O", "B-ENG", "B-ENG", "I-ENG"],
            lexicon={"un", "total", "de", "la"},
        )
        look, total_black = c.test.ents
        # "look" is not in the lexicon: not fully in lexicon
        self.assertFalse(look._.is_in_lexicon)
        # "total" is an ordinary Spanish word, "black" is not: mixed
        self.assertFalse(total_black._.is_in_lexicon)
        self.assertTrue(total_black._.is_partial_in_lexicon)

    def test_gazetteer(self):
        c = Cascabel.from_tokens(
            ["Un", "look", "total", "black"],
            ["O", "B-ENG", "B-ENG", "I-ENG"],
            gazetteer={"total black", "look"},
        )
        self.assertTrue(c.test.ents[0]._.is_in_gazetteer)
        self.assertTrue(c.test.ents[1]._.is_in_gazetteer)
        self.assertTrue(c.test[2]._.is_in_gazetteer)  # "total" tokenized
        self.assertFalse(c.test[0]._.is_in_gazetteer)  # "Un"


TINY_CONFIG = CharLMConfig(epochs=30, hidden_size=32, embedding_size=16,
                           min_label_spans=2)


class ShapeAnnotationTestCase(unittest.TestCase):
    def test_shape_without_training(self):
        c = Cascabel.from_tokens(["Un", "look"], ["O", "B-ENG"])
        self.assertIsNone(c.shape_models)
        ent = c.test.ents[0]
        self.assertIsNone(c.test[1]._.surprisal)
        for name in ("surprisal_profile", "surprisal_mean", "surprisal_spread",
                     "least_surprising_position", "prototypicality",
                     "label_fit", "closest_other_label"):
            self.assertIsNone(getattr(ent._, name), name)

    def test_shape_disabled(self):
        c = Cascabel.from_tokens(["Un", "look"], ["O", "B-ENG"],
                                 training=(["el", "look"], ["O", "B-ENG"]),
                                 shape=False)
        self.assertIsNone(c.shape_models)

    def test_surprisal_from_training(self):
        training = [(
            ["la", "casa", "de", "mi", "madre", "es", "grande", "y", "bonita",
             "pero", "queda", "lejos", "del", "centro", "streaming"],
            ["O"] * 14 + ["B-ENG"],
        )] * 3
        c = Cascabel.from_sentences(
            [(["me", "gusta", "el", "shopping", "casa"],
              ["O", "O", "O", "B-ENG", "O"])],
            training=training, charlm_config=TINY_CONFIG,
        )
        shopping, casa = c.test[3], c.test[4]
        self.assertGreater(shopping._.surprisal, casa._.surprisal)

    def test_surprisal_profile_summaries(self):
        training = [(["la", "casa", "de", "Ana", "Pi"],
                     ["O", "O", "O", "B-PER", "I-PER"])] * 3
        c = Cascabel.from_sentences(
            [(["la", "Zxqw", "casa"], ["O", "B-X", "I-X"])],
            training=training, charlm_config=TINY_CONFIG,
        )
        ent = c.test.ents[0]
        profile = ent._.surprisal_profile
        self.assertEqual(len(profile), 2)
        self.assertAlmostEqual(ent._.surprisal_mean, sum(profile) / 2)
        self.assertAlmostEqual(ent._.surprisal_spread,
                               max(profile) - min(profile))
        # "casa" is an ordinary training word: the least surprising token
        self.assertEqual(ent._.least_surprising_position, "last")

    def test_prototypicality_and_label_fit(self):
        training = [
            (["con", "Juan", "Pérez", "en", "Madrid"],
             ["O", "B-PER", "I-PER", "O", "B-LOC"]),
            (["con", "Ana", "López", "en", "Sevilla"],
             ["O", "B-PER", "I-PER", "O", "B-LOC"]),
        ] * 5
        c = Cascabel.from_sentences(
            [(["con", "Luis", "Pérez", "en", "Sevilla"],
              ["O", "B-PER", "I-PER", "O", "B-PER"])],
            training=training, charlm_config=TINY_CONFIG,
        )
        self.assertEqual(sorted(c.shape_models.label_models), ["LOC", "PER"])
        person, sevilla = c.test.ents
        self.assertIsNotNone(person._.prototypicality)
        self.assertEqual(person._.closest_other_label, "LOC")
        # "Sevilla" labelled PER looks like a LOC: negative label fit
        self.assertLess(sevilla._.label_fit, 0)
        self.assertGreater(person._.label_fit, sevilla._.label_fit)

    def test_rare_label_has_no_model(self):
        training = [(["con", "Juan", "en", "Madrid"],
                     ["O", "B-PER", "O", "B-LOC"])] * 3
        config = CharLMConfig(epochs=2, hidden_size=8, embedding_size=4,
                              min_label_spans=5)
        c = Cascabel.from_sentences(
            [(["con", "Ana"], ["O", "B-PER"])], training=training,
            charlm_config=config,
        )
        self.assertEqual(c.shape_models.label_models, {})
        self.assertIsNone(c.test.ents[0]._.prototypicality)
        self.assertIsNone(c.test.ents[0]._.label_fit)
        self.assertIsNotNone(c.test.ents[0]._.surprisal_mean)


if __name__ == "__main__":
    unittest.main()
