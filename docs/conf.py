# Sphinx configuration for cascabel documentation.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import cascabel  # noqa: E402

project = "cascabel"
author = "Elena Álvarez Mellado"
copyright = "2026, Elena Álvarez Mellado"
release = cascabel.__version__
version = cascabel.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_title = f"cascabel {release}"
