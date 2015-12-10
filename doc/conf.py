#!/usr/bin/env python3

import sys
import os
from unittest.mock import MagicMock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
]

MOCK_MODULES = [
    'yaml',
    'numpy',
    'matplotlib',
    'matplotlib.pyplot',
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

################################################################
# General
################################################################

project = 'Layered'
copyright = '2015, Danijar Hafner'
author = 'Danijar Hafner'
version = '0.1'
release = '0.1.4'
source_suffix = '.rst'
master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = ['_build']
pygments_style = 'sphinx'
add_module_names = False
todo_include_todos = False
language = None
htmlhelp_basename = 'Layereddoc'

################################################################
# HTML
################################################################

html_domain_indices = False
html_use_index = False
html_show_sphinx = False
html_show_copyright = False

################################################################
# Autodoc
################################################################

autoclass_content = 'class'
autodoc_member_order = 'bysource'
autodoc_default_flags = [
    'members',
    'undoc-members',
    'inherited-members',
    'show-inheritance',
]
autodoc_mock_imports = MOCK_MODULES


def autodoc_skip_member(app, what, name, obj, skip, options):
    keep = ['call', 'iter', 'getitem', 'setitem']
    if name.strip('_') in keep:
        return False
    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
