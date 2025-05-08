import os
from pathlib import Path
TYPE_CONFUSABLE_EXCLUDE = \
    ["la",
     "de"]


QUOTATIONS = ["’",
              "‘",
              "”",
              "“",
              "\"",
              "\'",
              "«",
              "»",
              ]

VALID_ONSETS = [
    "bl",
    "br",
    "pl",
    "pr",
    "cl",
    "cr",
    "gr",
    "gl",
    "fr",
    "fl",
    "tr",
    "dr",
    "ch",
    "qu",
    "gu",
    "rr",
    "ll"
]

VALID_CODAS = [
  "ns"
]

VOWELS = [
    "a",
    "e",
    "i",
    "o",
    "u",
    "á",
    "é",
    "í",
    "ó",
    "ú"
]

VALID_CONSONANT_ENDING = [
    "r",
    "s",
    "l",
    "n",
    "d",
    "z",
    "y"
]

VALID_TRANSITIONS = [
    "ct",
    "cc",
    "gn",
    "mp",
    "mb",
    "mn",
    "pt",
    "pc",
    "nn"
]

FORBIDDEN_PATTERNS = [
    "qua",
    "quo",
    "quu",
    "ze",
    "zi",
    "pca",
    "pco",
    "pcu",
    "ash",
    "ish",
    "osh",
    "ush",
    "ze",
    "zi",
    "sh"
]