import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('..'))

try:
    from importlib.metadata import version
    release = version('ebltable')
except Exception:
    release = 'unknown'

# Copy notebooks into docs/tutorials/ so nbsphinx can find them.
# The notebooks/ directory at the repo root is the source of truth;
# the copies here are generated and excluded from git.
_here = os.path.dirname(os.path.abspath(__file__))
_nb_src = os.path.join(_here, '..', 'notebooks')
_nb_dst = os.path.join(_here, 'tutorials')
os.makedirs(_nb_dst, exist_ok=True)
for _nb in os.listdir(_nb_src):
    if _nb.endswith('.ipynb'):
        shutil.copy2(os.path.join(_nb_src, _nb), os.path.join(_nb_dst, _nb))

project = 'ebltable'
copyright = '2024, Manuel Meyer'
author = 'Manuel Meyer'
version = release

extensions = [
    'nbsphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx_rtd_theme',
]

nbsphinx_execute = 'never'

autosummary_generate = True

autodoc_default_options = {
    'special-members': '__init__',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

intersphinx_cache_limit = 5

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
master_doc = 'index'

html_theme = 'sphinx_rtd_theme'
