"""Show how cascabel behaves on small, hand-made scenarios."""

from __future__ import annotations

import warnings

from cascabel import Cascabel

TRAIN = [
    (["Vivo", "en", "Madrid", "."], ["O", "O", "B-LOC", "O"]),
    (["Ganó", "el", "Madrid", "."], ["O", "O", "B-ORG", "O"]),
    (["La", "Comunidad", "de", "Madrid", "aprobó", "."], ["O", "B-ORG", "I-ORG", "I-ORG", "O", "O"]),
    (["Habló", "Sergei", "Yastrzhembsky", "."], ["O", "B-PER", "I-PER", "O"]),
    (["Lo", "publicó", "El", "País", "."], ["O", "O", "B-ORG", "I-ORG", "O"]),
    (["Viajó", "por", "el", "país", "."], ["O", "O", "O", "O", "O"]),
    (["SOCCER", "-", "JAPAN", "WIN"], ["O", "O", "B-LOC", "O"]),
    (["Llegó", "a", "Lyon", "."], ["O", "O", "B-LOC", "O"]),
    (["Ganó", "el", "Lyon", "."], ["O", "O", "B-ORG", "O"]),
    (["Compré", "una", "Apple", "."], ["O", "O", "O", "O"]),
    (["Visitó", "Bonn", "."], ["O", "B-ORG", "O"]),
]

TEST = [
    (["Vivo", "en", "Madrid", "."], ["O", "O", "B-LOC", "O"]),            # Madrid LOC
    (["Dijo", "Yastrzhembsky", "."], ["O", "B-PER", "O"]),                # seen only inside longer span
    (["Lo", "publicó", "El", "País", "."], ["O", "O", "B-ORG", "I-ORG", "O"]),
    (["Llegó", "a", "Japan", "."], ["O", "O", "B-LOC", "O"]),             # seen only as JAPAN
    (["Habló", "Lyon", "."], ["O", "B-PER", "O"]),                        # LOC/ORG in train, PER here
    (["Abrió", "Apple", "."], ["O", "B-ORG", "O"]),                       # seen only as O
    (["Visitó", "Bonn", "."], ["O", "B-LOC", "O"]),                       # seen only as ORG
    (["Firmó", "Juan", "Peña", "."], ["O", "B-PER", "I-PER", "O"]),       # entirely new
]

PRED = [
    (["Vivo", "en", "Madrid", "."], ["O", "O", "B-LOC", "O"]),            # correct
    (["Dijo", "Yastrzhembsky", "."], ["O", "O", "O"]),                     # missed
    (["Lo", "publicó", "El", "País", "."], ["O", "O", "O", "B-ORG", "O"]),  # overlap_missed
    (["Llegó", "a", "Japan", "."], ["O", "O", "B-LOC", "O"]),             # correct
    (["Habló", "Lyon", "."], ["O", "B-LOC", "O"]),                        # type
    (["Abrió", "Apple", "."], ["B-ORG", "I-ORG", "O"]),                   # overlap_spurious
    (["Visitó", "Bonn", "."], ["O", "B-ORG", "O"]),                       # type
    (["Firmó", "Juan", "Peña", "."], ["O", "B-PER", "B-PER", "O"]),       # split
]


def show_identity(c: Cascabel) -> None:
    print("\n== 1. Each test span: span layer, token layer, spelling variants ==")
    for ent in c.test.ents:
        words = ", ".join(f"{t.text}={t._.exposure}" for t in ent)
        print(f"\n{ent.text!r} ({ent.label_})")
        print(f"  span layer : identity={ent._.identity}  train_labels={ent._.train_labels}")
        print(f"  token layer: {words}")
        print(f"  variants   : {ent._.variant_labels}")


def show_summary(c: Cascabel) -> None:
    print("\n== 2. Radar identity axes ==")
    s = c.identity_summary("test")
    print(f"  new spans        {s['n_unseen']}/{s['n_spans']} = {s['unseen_share']:.0%}")
    print(f"  multi-label      {s['n_multi_label']}/{s['n_seen']} seen = {s['multi_label_share']:.0%}")
    print(f"  unseen label     {s['n_unseen_label']}/{s['n_seen']} seen = {s['unseen_label_share']:.0%}")
    print(f"  (extra) unseen but seen as spelling variant: {s['n_unseen_variant']}/{s['n_unseen']}")


def show_label_ambiguity(c: Cascabel) -> None:
    print("\n== 3. Label ambiguity inside training ==")
    amb = c.label_ambiguity("training")
    print(f"  {amb['n_multi_label']} of {amb['n_entities']} distinct training spans have >1 label")
    for e in amb["entities"]:
        if len(e["labels"]) > 1:
            print(f"  {e['text']!r}: {e['labels']}")


def show_errors(c: Cascabel) -> None:
    print("\n== 4. Predictions: error types ==")
    for a in c.alignments:
        gold = " | ".join(f"{s.text} {s.label_}" for s in a.gold) or "-"
        pred = " | ".join(f"{s.text} {s.label_}" for s in a.predicted) or "-"
        kind = a.error_type.value if a.error_type else "correct"
        print(f"  {kind:<17} gold: {gold:<22} pred: {pred}")
    print(f"  span F1 = {c.span_scores().f1:.2f}")


def show_surface(c: Cascabel) -> None:
    print("\n== 5. Surface attributes ==")
    for ent in c.test.ents:
        print(f"  {ent.text!r:<16} length={ent._.length} casing={ent._.casing} "
              f"sent_initial={ent._.is_sentence_initial}")


def show_bad_bio() -> None:
    print("\n== 6. Ill-formed BIO (stray I- after O) ==")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = Cascabel.from_sentences([(["en", "Nueva", "York"], ["O", "I-LOC", "I-LOC"])], shape=False)
    for w in caught:
        print(f"  warning: {w.message}")
    print(f"  read as: {[(e.text, e.label_) for e in c.test.ents]}")


def main() -> None:
    c = Cascabel.from_sentences(TEST, training=TRAIN, prediction=PRED, shape=False)
    show_identity(c)
    show_summary(c)
    show_label_ambiguity(c)
    show_errors(c)
    show_surface(c)
    show_bad_bio()


if __name__ == "__main__":
    main()
