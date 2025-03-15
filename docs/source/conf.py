# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import savant_rs

project = "savant_rs"
copyright = "2023, Ivan A. Kudriavtsev"
author = "Ivan A. Kudriavtsev"
release = savant_rs.version()
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

autosummary_generate = True
autosummary_imported_members = True

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]


# Custom theme options
html_theme_options = {
    'style_nav_header_background': '#2980B9',
    'style_external_links': True,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
