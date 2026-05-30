# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import sys, os, types
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath("../../.."), "src")))
sys.path.insert(1, os.path.abspath(os.path.join(os.path.abspath("../../.."), "docs")))
sys.path.append(os.path.abspath("sphinxext"))

autodock_mock_imports = ["numpy", "inspect", "pkgutil", "pathlib", "importlib"]

add_module_names = False

autodoc_default_options = {
    'members': True,
    # 'member-order': 'bysource',
    # 'special-members': '__init__',
    'undoc-members': True,
    # 'exclude-members': '__weakref__'
}

# -- Project information -----------------------------------------------------

project = "Scarabaeus"
copyright = "2025, ORCCA Laboratory, University of Colorado Boulder"
author = "ORCCA Laboratory, University of Colorado Boulder"

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = ""

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "sphinx_collections",
    "sphinx_gallery.load_style",
    "sphinx_tabs.tabs",
    "numpydoc",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxext.opengraph",
    # 'sphinx_autodoc_typehints',
]

autodoc_type_aliases = {'PCInfo': 'scarabaeus.utils.Constants.PCInfo',
                        'np.ndarray': 'numpy.ndarray'}

# set up for notebook tutorial rendering
nbsphinx_execute = 'auto'
nbsphinx_allow_errors = True

nbsphinx_thumbnails = {
    # basics
    '_collections/tutorials/basics_AWU_and_AWF':
        '_images/_collections_tutorials_basics_AWU_and_AWF_22_0.png',
    '_collections/tutorials/basics_EpochArray':
        '_images/_collections_tutorials_basics_EpochArray_23_0.png',
    '_collections/tutorials/basics_SpiceManager_and_CelestialBodies':
        '_images/_collections_tutorials_basics_SpiceManager_and_CelestialBodies_17_1.png',
    # intermediate
    '_collections/tutorials/intermediate_Propagator_and_Trajectory':
        '_images/_collections_tutorials_intermediate_Propagator_and_Trajectory_21_1.png',
    '_collections/tutorials/intermediate_nPlateModel_and_Attitude':
        '_images/_collections_tutorials_intermediate_nPlateModel_and_Attitude_12_0.png',
    '_collections/tutorials/intermediate_Measurements':
        '_images/_collections_tutorials_intermediate_Measurements_22_1.png',
    # advanced
    '_collections/tutorials/advanced_IdealMSR_BatchOD':
        '_images/_collections_tutorials_advanced_IdealMSR_BatchOD_35_0.png',
    '_collections/tutorials/advanced_IdealMSR_SequentialOD':
        '_images/_collections_tutorials_advanced_IdealMSR_SequentialOD_10_0.png',
    '_collections/tutorials/advanced_RealMSR_MediaCorrections':
        '_images/_collections_tutorials_advanced_RealMSR_MediaCorrections_9_0.png',
    '_collections/tutorials/advanced_RealMSR_OSIRIS_REx_OD':
        '_images/_collections_tutorials_advanced_RealMSR_OSIRIS_REx_OD_14_0.png',
}

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
tutorials_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../tutorials/"))
template_class_path = str(Path(__file__).parent.parent.parent / 'TemplateClass.py')
collections = {'tutorial_notebooks' : {'driver': 'copy_folder',
                                       'source': tutorials_path,
                                       'target': 'tutorials'},
               'template_class'     : {'driver': 'copy_file',
                                       'source': template_class_path,
                                       'target': 'TemplateClass.py'}}
template_class_path = str(Path(__file__).parent.parent.parent / 'TemplateClass.py')
collections = {'tutorial_notebooks' : {'driver': 'copy_folder',
                                       'source': tutorials_path,
                                       'target': 'tutorials'},
               'template_class'     : {'driver': 'copy_file',
                                       'source': template_class_path,
                                       'target': 'TemplateClass.py'}}

napoleon_use_param = True
napoleon_preprocess_types = True
# typehints_fully_qualified = False
autosummary_generate = False

python_use_unqualified_type_names = True
add_module_names = False

# for opengraph
ogp_site_url = "https://ccar-orcca.github.io/scarabaeus-docs/index.html"
ogp_image = "https://ccar-orcca.github.io/scarabaeus-docs/_images/scarabaeus_logo.png"
ogp_description_length = 500

autodoc_member_order = "groupwise"
autodoc_typehints = "signature"

# make autodoc read the pyi stub instead of the compiled module
_stub_path = Path(__file__).parent.parent.parent.parent / "src" / "scarabaeus" / "scarabaeus_rust.pyi"
_mod = types.ModuleType("scarabaeus.scarabaeus_rust")
_mod.__file__ = str(_stub_path)
exec(compile(_stub_path.read_text(), _stub_path, "exec"), _mod.__dict__)
sys.modules["scarabaeus.scarabaeus_rust"] = _mod
sys.modules["scarabaeus_rust"] = _mod

# for constants table
sys.path.insert(0, os.path.abspath('../../src'))

def generate_constants_table(app):
    from scarabaeus.utils.Constants import Constants, PCInfo
    from scarabaeus import ArrayWUnits

    major_bodies = ['SUN', 'MERCURY', 'VENUS', 'EARTH', 'MARS',
                    'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE']

    def make_body_table(body_names):
        lines = []
        lines.append('.. list-table::')
        lines.append('   :header-rows: 1')
        lines.append('   :widths: 15 10 15 15 15 15 15')
        lines.append('')
        lines.append('   * - Body')
        lines.append('     - SPICE ID')
        lines.append('     - Mass (kg)')
        lines.append('     - GM (km³/s²)')
        lines.append('     - Mean Radius (km)')
        lines.append('     - Eq. Radius (km)')
        lines.append('     - Primary')

        for name in body_names:
            val = getattr(Constants, name, None)
            if val is None or not isinstance(val, PCInfo):
                continue
            eq_radius = f"{val.equatorial_radius.values:.4g}" if val.equatorial_radius is not None else '—'
            primary = val.primary_body if val.primary_body else '—'
            lines.append(f'   * - {val.name}')
            lines.append(f'     - {val.body_center_id}')
            lines.append(f'     - {val.mass.values:.4e}')
            lines.append(f'     - {val.GM.values:.6g}')
            lines.append(f'     - {val.mean_radius.values}')
            lines.append(f'     - {eq_radius}')
            lines.append(f'     - {primary}')

        return '\n'.join(lines)

    def make_physical_table():
        lines = []
        lines.append('.. list-table::')
        lines.append('   :header-rows: 1')
        lines.append('   :widths: 35 35 30')
        lines.append('')
        lines.append('   * - Name')
        lines.append('     - Value')
        lines.append('     - Units')

        for attr_name in dir(Constants):
            if attr_name.startswith('_'):
                continue
            val = getattr(Constants, attr_name)
            if not isinstance(val, ArrayWUnits):
                continue
            units = str(val.units) if val.units is not None else '—'
            lines.append(f'   * - ``{attr_name}``')
            lines.append(f'     - {val.values}')
            lines.append(f'     - {units}')

        return '\n'.join(lines)

    all_pcinfo = [name for name in dir(Constants)
                  if isinstance(getattr(Constants, name), PCInfo)]
    minor_bodies = [name for name in all_pcinfo if name not in major_bodies]

    output = 'Physical Constants\n^^^^^^^^^^^^^^^^^^\n\n'
    output += make_physical_table()
    output += '\n\nMajor Bodies\n^^^^^^^^^^^^^\n\n'
    output += make_body_table(major_bodies)
    output += '\n\nMinor Bodies\n^^^^^^^^^^^^^^^^^^^^^\n\n'
    output += make_body_table(minor_bodies)

    output_path = os.path.join(os.path.dirname(__file__),
                               'api_ref/utils/planetary_constants_table.rst.inc')
    with open(output_path, 'w') as f:
        f.write(output)

def setup(app):
    app.connect('builder-inited', generate_constants_table)

# lets links to types from the specified libraries become clickable
# see these links for more links with intersphinx
# https://gist.github.com/bskinn/0e164963428d4b51017cebdb6cda5209
# https://stackoverflow.com/questions/21538983/specifying-targets-for-intersphinx-links-to-numpy-scipy-and-matplotlib
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.11/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# sphinx doesn't like long python blocks with a lot of stuff in it -> disable the warning for it
# since it still renders correctly. all this warning does is fill the terminal with a bunch of red text
suppress_warnings = ["misc.highlighting_failure",
                     "docutils",
                     "ref.citation",
                     "myst.xref_missing",
                     "ref.ref",
                     'toc.multiple',
                     'toc',
                     'autodoc',
                     'py.duplicate-object']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"

numpydoc_attributes_as_param_list = True
numpydoc_use_plots = True
numpydoc_class_members_toctree = False

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "show_toc_level": 5,
    "show_nav_level": 2,
    "navigation_depth": -1,
    "primary_sidebar_end": ["indices.html"],
    "logo": {"text": "Scarabaeus"},
    "header_links_before_dropdown": 6,
}

html_context = {"default_mode": "dark"}
html_show_sourcelink = False

# Specifies logo image to display in upper left
html_logo = os.path.join(os.path.abspath("../.."), "images", "scarabaeus_logo.png")

# Specifies favicon image
# html_favicon = os.path.join(os.path.abspath('../..'), 'images', 'ORCCA_logo.ico')
# html_favicon = "https://raw.githubusercontent.com/CCAR-ORCCA/scarabaeus-docs/refs/heads/main/_images/ORCCA_logo.ico"
html_favicon = '_static/favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# html_static_path = ['../_build/html/_static']
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ['custom.js']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "Scarabaeusdoc"


# -- Options for LaTeX output ------------------------------------------------
# see link below for more
# https://www.sphinx-doc.org/en/master/latex.html

# latex_additional_files = ['latex_style.tex.txt']

# latex_elements = {
#     'preamble': r'\\input{latex_style.tex.txt}'
# }

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "Scarabaeus.tex",
        "Scarabaeus Documentation",
        "ORCCA Laboratory, University of Colorado Boulder",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "scarabaeus", "Scarabaeus Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Scarabaeus",
        "Scarabaeus Documentation",
        author,
        "Scarabaeus",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

# extensions = [
#    "sphinxcontrib.tikz",
# ]

# tikz_proc_suite = "pdf2svg"  # or 'dvisvgm', 'imagemagick', depending on your setup
# tikz_use_preview = True
