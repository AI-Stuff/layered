#!/usr/bin/env python3

import sys
import os
import mock


sys.path.insert(0, os.path.abspath('../layered'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
]

MOCK_MODULES = [
    'yaml',
    'numpy',
    'matplotlib.pyploy',
    'matplotlib.cbook',
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

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

################################################################
# HTML
################################################################

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
    keep = ['init']
    if name.strip('_') in keep:
        return False
    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
