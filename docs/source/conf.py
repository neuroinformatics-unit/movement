# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import setuptools_scm

# Used when building API docs, put the dependencies
# of any class you are documenting here
autodoc_mock_imports = []

# Add the module path to sys.path here.
# If the directory is relative to the documentation root,
# use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../.."))

project = "movement"
copyright = "2023, University College London"
author = "Niko Sirmpilatze"
try:
    release = setuptools_scm.get_version(root="../..", relative_to=__file__)
    release = release.split("+")[0]  # remove git hash
except LookupError:
    # if git is not initialised, still allow local build
    # with a dummy version
    release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "notfound.extension",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_sitemap",
    "sphinx.ext.autosectionlabel",
    "ablog",
]

# Configure the myst parser to enable cool markdown features
# See https://sphinx-design.readthedocs.io
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
# Automatically add anchors to markdown headings
myst_heading_anchors = 4

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Automatically generate stub pages for API
autosummary_generate = True
autodoc_default_flags = ["members", "inherited-members"]

# Prefix section labels with the document name
autosectionlabel_prefix_document = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "**.ipynb_checkpoints",
    # to ensure that include files (partial pages) aren't built, exclude them
    # https://github.com/sphinx-doc/sphinx/issues/1965#issuecomment-124732907
    "**/includes/**",
    # exclude .py and .ipynb files in examples generated by sphinx-gallery
    # this is to prevent sphinx from complaining about duplicate source files
    "examples/*.ipynb",
    "examples/*.py",
]

# Configure Sphinx gallery
sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "filename_pattern": "/*.py",  # which files to execute before inclusion
    "gallery_dirs": ["examples"],  # output directory
    "run_stale_examples": True,  # re-run examples on each build
    # Integration with Binder, see https://sphinx-gallery.github.io/stable/configuration.html#generate-binder-links-for-gallery-notebooks-experimental
    "binder": {
        "org": "neuroinformatics-unit",
        "repo": "movement",
        "branch": "gh-pages",
        "binderhub_url": "https://mybinder.org",
        "dependencies": ["environment.yml"],
    },
    "reference_url": {"movement": None},
    "default_thumb_file": "source/_static/data_icon.png",  # default thumbnail image
    "remove_config_comments": True,
    # do not render config params set as # sphinx_gallery_config [= value]
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_title = "movement"

# Customize the theme
html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/neuroinformatics-unit/movement",
            # Icon class (if "type": "fontawesome"),
            # or path to local image (if "type": "local")
            "icon": "fa-brands fa-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            "name": "Zulip (chat)",
            "url": "https://neuroinformatics.zulipchat.com/#narrow/stream/406001-Movement",
            "icon": "fa-solid fa-comments",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "text": f"{project} v{release}",
    },
    "footer_start": ["footer_start"],
    "footer_end": ["footer_end"],
    "external_links": [],
}

# Redirect the webpage to another URL
# Sphinx will create the appropriate CNAME file in the build directory
# The default is the URL of the GitHub pages
# https://www.sphinx-doc.org/en/master/usage/extensions/githubpages.html
github_user = "neuroinformatics-unit"
html_baseurl = "https://movement.neuroinformatics.dev"
sitemap_url_scheme = "{link}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    ("css/custom.css", {"priority": 100}),
]
html_favicon = "_static/light-logo-niu.png"

# The linkcheck builder will skip verifying that anchors exist when checking
# these URLs
linkcheck_anchors_ignore_for_url = [
    "https://gin.g-node.org/G-Node/Info/wiki/",
    "https://neuroinformatics.zulipchat.com/",
    "https://github.com/talmolab/sleap/blob/v1.3.3/sleap/info/write_tracking_h5.py",
]
# A list of regular expressions that match URIs that should not be checked
linkcheck_ignore = [
    "https://pubs.acs.org/doi/*",  # Checking dois is forbidden here
    "https://opensource.org/license/bsd-3-clause/",  # to avoid odd 403 error
]

myst_url_schemes = {
    "http": None,
    "https": None,
    "ftp": None,
    "mailto": None,
    "movement-github": "https://github.com/neuroinformatics-unit/movement/{{path}}",
    "movement-zulip": "https://neuroinformatics.zulipchat.com/#narrow/stream/406001-Movement",
    "conda": "https://docs.conda.io/en/latest/",
    "dlc": "https://www.mackenziemathislab.org/deeplabcut/",
    "gin": "https://gin.g-node.org/{{path}}#{{fragment}}",
    "github-docs": "https://docs.github.com/en/{{path}}#{{fragment}}",
    "mamba": "https://mamba.readthedocs.io/en/latest/",
    "myst-parser": "https://myst-parser.readthedocs.io/en/latest/{{path}}#{{fragment}}",
    "napari": "https://napari.org/dev/{{path}}",
    "setuptools-scm": "https://setuptools-scm.readthedocs.io/en/latest/{{path}}#{{fragment}}",
    "sleap": "https://sleap.ai/{{path}}#{{fragment}}",
    "sphinx-doc": "https://www.sphinx-doc.org/en/master/usage/{{path}}#{{fragment}}",
    "sphinx-gallery": "https://sphinx-gallery.github.io/stable/{{path}}#{{fragment}}",
    "xarray": "https://docs.xarray.dev/en/stable/{{path}}#{{fragment}}",
    "lp": "https://lightning-pose.readthedocs.io/en/stable/{{path}}#{{fragment}}",
    "via": "https://www.robots.ox.ac.uk/~vgg/software/via/{{path}}#{{fragment}}",
}

intersphinx_mapping = {
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}


# What to show on the 404 page
notfound_context = {
    "title": "Page Not Found",
    "body": """
<h1>Page Not Found</h1>

<p>Sorry, we couldn't find that page.</p>

<p>We occasionally restructure the movement website, and some links may have broken.</p> 

<p>Try using the search box or go to the homepage.</p>
""",
}

# needed for GH pages (vs readthedocs),
# because we have no '/<language>/<version>/' in the URL
notfound_urls_prefix = None
