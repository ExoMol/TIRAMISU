# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import re
import sys

sys.path.insert(0, os.path.abspath('..'))
# Building the docs only imports the package and runs no parallel kernels, so it needs no TBB (tiramisu.config's
# default threading layer): numba's built-in workqueue layer serves.
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')

project = 'TIRAMISU'
copyright = '2025, Charles A. Bowesman'
author = 'Charles A. Bowesman'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_parser',
]

# API reference: one page per module (api.rst, templates in _templates/autosummary), public members only.
autosummary_generate = True
autodoc_member_order = 'bysource'
autoclass_content = 'both'  # several classes document their parameters in __init__
autodoc_typehints = 'description'
typehints_fully_qualified = False
typehints_defaults = "braces"
autodoc_default_options = {
    'members': True,
}
napoleon_google_docstring = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# Markdown pages, with $...$ and $$...$$ maths.
myst_enable_extensions = ['dollarmath', 'amsmath']

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'furo'
html_title = 'TIRAMISU'
html_static_path = ['_static']
html_js_files = ['dark_default.js']  # Furo follows the system theme; start in dark mode unless the reader chose one.

# Docstrings cite references as doi:10.xxx/... and arXiv:NNNN.NNNNN (plain text, so they also read well in the source and
# in docutils' LaTeX output); link them in the rendered docs only.
_DOI = re.compile(r'(?<![\w/`<])doi:\s?(10\.\d{4,9}/[^\s`<>]+)')
_ARXIV = re.compile(r'(?<![\w/`<])arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)')


def _link(text: str, identifier: str, url: str) -> str:
    # Trailing punctuation and unbalanced closing parentheses belong to the sentence, not the identifier.
    trail = ''
    while identifier and (identifier[-1] in '.,;:' or (identifier[-1] == ')' and
                                                         identifier.count('(') < identifier.count(')'))):
        trail, identifier = identifier[-1] + trail, identifier[:-1]
    return f'`{text}{identifier} <{url}{identifier}>`__{trail}'


def _link_identifiers(app, what, name, obj, options, lines):
    for i, line in enumerate(lines):
        line = _DOI.sub(lambda m: _link('doi:', m.group(1), 'https://doi.org/'), line)
        lines[i] = _ARXIV.sub(lambda m: _link('arXiv:', m.group(1), 'https://arxiv.org/abs/'), line)


def setup(app):
    app.connect('autodoc-process-docstring', _link_identifiers)
