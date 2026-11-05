"""Language-agnostic constants used across the library."""

#: Characters treated as quotation marks when detecting quoted entities.
QUOTATION_MARKS = frozenset(
    [
        "’",  # right single quotation mark
        "‘",  # left single quotation mark
        "”",  # right double quotation mark
        "“",  # left double quotation mark
        '"',
        "'",
        "«",  # «
        "»",  # »
    ]
)
