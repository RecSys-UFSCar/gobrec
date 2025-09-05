# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'GOBRec'
copyright = '2025, Gregorio'
author = 'Gregorio'
release = '0.4.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",   # needed for autodoc directives
    "sphinx.ext.napoleon",  # optional, works with numpydoc too
    "sphinx.ext.viewcode",  # optional, adds source code links
    "numpydoc",             # the numpydoc extension
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "navbar_end": ["search-field.html", "navbar-icon-links.html"],
}
html_baseurl = "https://recsys-ufscar.github.io/gobrec/"
html_static_path = ['_static']
html_css_files = []
html_js_files = []
html_use_relative_paths = True
html_theme_options = {
    "navigation_with_keys": True,
}

html_context = {
    "display_github": True,
    "github_user": "recsys-ufscar",
    "github_repo": "gobrec",
    "github_version": "main",
    "conf_py_path": "/docs/",
}



def setup(app):
    app.connect("build-finished", create_nojekyll)

def create_nojekyll(app, exception):
    if app.builder.name == "html":
        nojekyll_path = os.path.join(app.outdir, ".nojekyll")
        with open(nojekyll_path, "w") as f:
            f.write("")