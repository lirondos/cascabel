"""Command-line interface: ``cascabel`` / ``python -m cascabel``.

Examples::

    cascabel test.conll pred.conll
    cascabel test.conll pred.conll --training train.conll --format html -o report.html
    cascabel test.conll --training train.conll        # dataset profile only
    cascabel test.spacy pred.spacy --spacy
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cascabel",
        description=(
            "Span profiling and fine-grained error analysis for NER and span-labeling datasets and models. "
            "Inputs are CoNLL column files (token first column, BIO tag last "
            "column, blank line between sentences) or spaCy DocBin files "
            "with --spacy."
        ),
    )
    parser.add_argument("test", help="gold-annotated evaluation file")
    parser.add_argument(
        "prediction",
        nargs="?",
        default=None,
        help="prediction file, token-aligned with the test file "
        "(omit for a dataset profile only)",
    )
    parser.add_argument("--training", help="training file (enables seen/unseen, surprisal and prototypicality)")
    parser.add_argument("--lexicon", help="background lexicon file, one word per line")
    parser.add_argument("--gazetteer", help="gazetteer file, one entity per line")
    parser.add_argument(
        "--spacy",
        action="store_true",
        help="treat input files as serialized spaCy DocBin (.spacy) files",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="CoNLL column delimiter (default: any whitespace)",
    )
    parser.add_argument(
        "--scheme",
        default="BIO",
        help="tagging scheme of the input files: BIO (default), IOB1 "
        "(original CoNLL-2003 files) or BIOES/BILOU",
    )
    parser.add_argument(
        "--no-shape",
        action="store_true",
        help="skip training the character language models (surprisal, prototypicality)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("text", "json", "html", "card"),
        default="text",
        help="output format (default: text); 'card' renders a standardized "
        "markdown dataset-profile or system-output card",
    )
    parser.add_argument("--dataset-name", default=None,
                        help="dataset name shown on the card")
    parser.add_argument("--system-name", default=None,
                        help="system name shown on the card")
    parser.add_argument(
        "-o",
        "--output",
        help="write the report to a file instead of stdout",
    )
    parser.add_argument(
        "--all-sentences",
        action="store_true",
        help="HTML only: include sentences without errors in the sentence view",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + _version(),
    )
    return parser


def _version() -> str:
    from cascabel import __version__

    return __version__


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Windows consoles often default to a legacy codepage (cp1252) that
    # cannot encode the report's typographic characters.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass

    from cascabel import Cascabel

    common = {
        "training": args.training,
        "prediction": args.prediction,
        "lexicon": args.lexicon,
        "gazetteer": args.gazetteer,
        "shape": not args.no_shape,
    }
    try:
        if args.spacy:
            casc = Cascabel.from_spacy_binary(args.test, **common)
        else:
            conll_options = {"delimiter": args.delimiter} if args.delimiter else {}
            casc = Cascabel.from_conll(
                args.test, conll_options=conll_options, scheme=args.scheme, **common
            )
    except (OSError, ValueError) as error:
        print(f"cascabel: error: {error}", file=sys.stderr)
        return 2

    if args.format == "text":
        output = casc.summary()
    elif args.format == "json":
        output = json.dumps(casc.report(), indent=2, ensure_ascii=False)
    elif args.format == "card":
        output = casc.card(dataset=args.dataset_name or args.test,
                           system=args.system_name or args.prediction)
    else:  # html
        output = casc.to_html(only_errors=not args.all_sentences)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
        print(f"report written to {args.output}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
