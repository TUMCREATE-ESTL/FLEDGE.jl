# Configuration file for the Sphinx documentation builder.
# - Documentation: <http://www.sphinx-doc.org/en/master/config>

# Project information.
project = 'FLEDGE'
copyright = '2018-2021, TUMCREATE'
author = 'TUMCREATE'

# Extensions.
# - Add any Sphinx extension module names here, as strings. They can be
#   extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
#   ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx_markdown_tables',
    'sphinx.ext.mathjax',
    'recommonmark',
    'sphinx_multiversion'
]

# Extension settings.
# - sphinx.ext.autodoc: <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>
# - sphinx.ext.napoleon: <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>
autodoc_default_options = {
    'members': None,
    'show-inheritance': None,
    'member-order': 'bysource'
}
autodoc_typehints = 'description'
autodoc_mock_imports = [
    # Please note: Do not remove deprecated dependencies, because these are still needed for docs of previous versions.
    'cvxpy',
    'cobmo',
    'cv2',
    'diskcache',
    'dill',
    'kaleido',
    'matplotlib',
    'multimethod',
    'multiprocess',  # Deprecated.
    'networkx',
    'natsort',
    'numpy',
    'opendssdirect',
    'pandas',
    'plotly',
    'pyomo',  # Deprecated.
    'ray',
    'scipy',
]
napoleon_use_ivar = True

# Source settings.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
master_doc = 'index'

# Exclude settings.
# - List of patterns, relative to source directory, that match files and
#   directories to ignore when looking for source files.
#   This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md']

# HTML theme settings.
# - The theme to use for HTML and HTML Help pages.  See the documentation for
#   a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
html_favicon = 'assets/favicon.ico'
templates_path = ['templates']

# Sphinx multiversion settings.
# - Explicitly include all branches, tags from all remotes.
smv_tag_whitelist = r'^.*$'
smv_branch_whitelist = r'^.*$'
smv_remote_whitelist = r'^.*$'

# Recommonmark settings.
# - Documentation: <https://recommonmark.readthedocs.io/en/latest/auto_structify.html>
from recommonmark.transform import AutoStructify
def setup(app):
    app.add_transform(AutoStructify)
