"""
Notes:
    Based on template code in: ~/code/kwimage/docs/source/conf.py

    http://docs.readthedocs.io/en/latest/getting_started.html

    pip install sphinx sphinx-autobuild sphinx_rtd_theme sphinxcontrib-napoleon

    cd ~/code/kwimage
    mkdir docs
    cd docs

    sphinx-quickstart

    # need to edit the conf.py

    cd ~/code/kwimage/docs
    sphinx-apidoc -f -o ~/code/kwimage/docs/source ~/code/kwimage/kwimage --separate
    make html

    Also:
        To turn on PR checks

        https://docs.readthedocs.io/en/stable/guides/autobuild-docs-for-pull-requests.html

        https://readthedocs.org/dashboard/kwimage/advanced/

        ensure your github account is connected to readthedocs
        https://readthedocs.org/accounts/social/connections/

        ### For gitlab

        The user will need to enable the repo on their readthedocs account:
        https://readthedocs.org/dashboard/import/manual/?

        To enable the read-the-docs go to https://readthedocs.org/dashboard/ and login

        Make sure you have a .readthedocs.yml file

        Click import project: (for github you can select, but gitlab you need to import manually)
            Set the Repository NAME: $REPO_NAME
            Set the Repository URL: $REPO_URL

        For gitlab you also need to setup an integrations and add gitlab
        incoming webhook Then go to $REPO_URL/hooks and add the URL

        Will also need to activate the main branch:
            https://readthedocs.org/projects/kwimage/versions/
"""
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import sphinx_rtd_theme
from os.path import exists
from os.path import dirname
from os.path import join


def parse_version(fpath):
    """
    Statically parse the version number from a python file
    """
    import ast
    if not exists(fpath):
        raise ValueError('fpath={!r} does not exist'.format(fpath))
    with open(fpath, 'r') as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, 'id', None) == '__version__':
                    self.version = node.value.s
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version

project = 'kwimage'
copyright = '2022, Jon Crall'
author = 'Jon Crall'
modname = 'kwimage'

modpath = join(dirname(dirname(dirname(__file__))), modname, '__init__.py')
release = parse_version(modpath)
version = '.'.join(release.split('.')[0:2])


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    # 'myst_parser',  # TODO
]

todo_include_todos = True
napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True

autodoc_inherit_docstrings = False

autodoc_member_order = 'bysource'
# autodoc_mock_imports = ['torch', 'torchvision', 'visdom']

intersphinx_mapping = {
    # 'pytorch': ('http://pytorch.org/docs/master/', None),
    'python': ('https://docs.python.org/3', None),
    'click': ('https://click.palletsprojects.com/', None),
    # 'xxhash': ('https://pypi.org/project/xxhash/', None),
    # 'pygments': ('https://pygments.org/docs/', None),
    # 'tqdm': ('https://tqdm.github.io/', None),
    # Requries that the repo have objects.inv
    'kwarray': ('https://kwarray.readthedocs.io/en/latest/', None),
    'kwimage': ('https://kwimage.readthedocs.io/en/latest/', None),
    # 'kwplot': ('https://kwplot.readthedocs.io/en/latest/', None),
    'ndsampler': ('https://ndsampler.readthedocs.io/en/latest/', None),
    'ubelt': ('https://ubelt.readthedocs.io/en/latest/', None),
    'xdoctest': ('https://xdoctest.readthedocs.io/en/latest/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'scriptconfig': ('https://scriptconfig.readthedocs.io/en/latest/', None),

}
__dev_note__ = """
python -m sphinx.ext.intersphinx https://docs.python.org/3/objects.inv
python -m sphinx.ext.intersphinx https://kwcoco.readthedocs.io/en/latest/objects.inv
python -m sphinx.ext.intersphinx https://networkx.org/documentation/stable/objects.inv
python -m sphinx.ext.intersphinx https://kwarray.readthedocs.io/en/latest/objects.inv
python -m sphinx.ext.intersphinx https://kwimage.readthedocs.io/en/latest/objects.inv
python -m sphinx.ext.intersphinx https://ubelt.readthedocs.io/en/latest/objects.inv
python -m sphinx.ext.intersphinx https://networkx.org/documentation/stable/objects.inv
"""


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    # 'logo_only': True,
}
# html_logo = '.static/kwimage.svg'
# html_favicon = '.static/kwimage.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

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
htmlhelp_basename = 'kwimagedoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'kwimage.tex', 'kwimage Documentation',
     'Jon Crall', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'kwimage', 'kwimage Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'kwimage', 'kwimage Documentation',
     author, 'kwimage', 'One line description of project.',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------


from sphinx.domains.python import PythonDomain  # NOQA
# from sphinx.application import Sphinx  # NOQA
from typing import Any, List  # NOQA


class PatchedPythonDomain(PythonDomain):
    """
    References:
        https://github.com/sphinx-doc/sphinx/issues/3866
    """
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        # TODO: can use this to resolve references nicely
        # if target.startswith('ub.'):
        #     target = 'ubelt.' + target[3]
        return_value = super(PatchedPythonDomain, self).resolve_xref(
            env, fromdocname, builder, typ, target, node, contnode)
        return return_value


class GoogleStyleDocstringProcessor:
    """
    A small extension that runs after napoleon and reformats erotemic-flavored
    google-style docstrings for sphinx.
    """

    def __init__(self, autobuild=1):
        self.registry = {}
        if autobuild:
            self._register_builtins()

    def register_section(self, tag, alias=None):
        """
        Decorator that adds a custom processing function for a non-standard
        google style tag. The decorated function should accept a list of
        docstring lines, where the first one will be the google-style tag that
        likely needs to be replaced, and then return the appropriate sphinx
        format (TODO what is the name? Is it just RST?).
        """
        alias = [] if alias is None else alias
        alias = [alias] if not isinstance(alias, (list, tuple, set)) else alias
        alias.append(tag)
        alias = tuple(alias)
        # TODO: better tag patterns
        def _wrap(func):
            self.registry[tag] = {
                'tag': tag,
                'alias': alias,
                'func': func,
            }
            return func
        return _wrap

    def _register_builtins(self):
        """
        Adds definitions I like of CommandLine, TextArt, and Ignore
        """

        @self.register_section(tag='CommandLine')
        def commandline(lines):
            new_lines = []
            new_lines.append('.. rubric:: CommandLine')
            new_lines.append('')
            new_lines.append('.. code-block:: bash')
            new_lines.append('')
            new_lines.extend(lines[1:])
            return new_lines

        @self.register_section(tag='SpecialExample', alias=['Benchmark', 'Sympy', 'Doctest'])
        def benchmark(lines):
            import textwrap
            new_lines = []
            tag = lines[0].replace(':', '').strip()
            # new_lines.append(lines[0])  # TODO: it would be nice to change the tagline.
            # new_lines.append('')
            new_lines.append('.. rubric:: {}'.format(tag))
            new_lines.append('')
            new_text = textwrap.dedent('\n'.join(lines[1:]))
            redone = new_text.split('\n')
            new_lines.extend(redone)
            # import ubelt as ub
            # print('new_lines = {}'.format(ub.repr2(new_lines, nl=1)))
            # new_lines.append('')
            return new_lines

        @self.register_section(tag='TextArt', alias=['Ascii'])
        def text_art(lines):
            new_lines = []
            new_lines.append('.. rubric:: TextArt')
            new_lines.append('')
            new_lines.append('.. code-block:: bash')
            new_lines.append('')
            new_lines.extend(lines[1:])
            return new_lines

        @self.register_section(tag='Ignore')
        def ignore(lines):
            return []

    def process(self, lines):
        """
        Example:
            >>> self = GoogleStyleDocstringProcessor()
            >>> lines = ub.codeblock(
            ...     '''
            ...     Hello world
            ...
            ...     CommandLine:
            ...         hi
            ...
            ...     CommandLine:
            ...
            ...         bye
            ...
            ...     TextArt:
            ...
            ...         1
            ...         2
            ...
            ...         345
            ...
            ...     Foobar:
            ...
            ...     TextArt:
            ...     ''').split(chr(10))
            >>> new_lines = self.process(lines[:])
            >>> print(chr(10).join(new_lines))
        """
        orig_lines = lines[:]
        new_lines = []
        curr_mode = '__doc__'
        accum = []

        def accept():
            """ called when we finish reading a section """
            if curr_mode == '__doc__':
                # Keep the lines as-is
                new_lines.extend(accum)
            else:
                # Process this section with the given function
                regitem = self.registry[curr_mode]
                func = regitem['func']
                fixed = func(accum)
                new_lines.extend(fixed)
            # Reset the accumulator for the next section
            accum[:] = []

        for line in orig_lines:

            found = None
            for regitem in self.registry.values():
                if line.startswith(regitem['alias']):
                    found = regitem['tag']
                    break
            if not found and line and not line.startswith(' '):
                # if the line startswith anything but a space, we are no longer
                # in the previous nested scope. NOTE: This assumption may not
                # be general, but it works for my code.
                found = '__doc__'

            if found:
                # New section is found, accept the previous one and start
                # accumulating the new one.
                accept()
                curr_mode = found

            accum.append(line)

        # Finialize the last section
        accept()

        lines[:] = new_lines
        # make sure there is a blank line at the end
        if lines and lines[-1]:
            lines.append('')

        return lines

    def process_docstring_callback(self, app, what_: str, name: str, obj: Any,
                                   options: Any, lines: List[str]) -> None:
        """
        Callback to be registered to autodoc-process-docstring

        Custom process to transform docstring lines Remove "Ignore" blocks

        Args:
            app (sphinx.application.Sphinx): the Sphinx application object

            what (str):
                the type of the object which the docstring belongs to (one of
                "module", "class", "exception", "function", "method", "attribute")

            name (str): the fully qualified name of the object

            obj: the object itself

            options: the options given to the directive: an object with
                attributes inherited_members, undoc_members, show_inheritance
                and noindex that are true if the flag option of same name was
                given to the auto directive

            lines (List[str]): the lines of the docstring, see above

        References:
            https://www.sphinx-doc.org/en/1.5.1/_modules/sphinx/ext/autodoc.html
            https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
        """
        # print(f'name={name}')
        # print('BEFORE:')
        # import ubelt as ub
        # print('lines = {}'.format(ub.repr2(lines, nl=1)))

        self.process(lines)

        # docstr = '\n'.join(lines)
        # if 'Convert the Mask' in docstr:
        #     import xdev
        #     xdev.embed()

        # if 'keys in this dictionary ' in docstr:
        #     import xdev
        #     xdev.embed()

        if 1:
            # DEVELOPING
            if any('REQUIRES(--show)' in line for line in lines):
                # import xdev
                # xdev.embed()
                create_doctest_figure(app, obj, name, lines)

        # FORMAT THE RETURNS SECTION A BIT NICER
        # Split by sphinx types
        import re
        tag_pat = re.compile(r'^:(\w*):')
        directive_pat = re.compile(r'^.. (\w*)::\s*(\w*)')
        sphinx_parts = []
        for idx, line in enumerate(lines):
            tag_match = tag_pat.search(line)
            directive_match = directive_pat.search(line)
            if tag_match:
                tag = tag_match.groups()[0]
                sphinx_parts.append({
                    'tag': tag, 'start_offset': idx,
                    'type': 'tag',
                })
            elif directive_match:
                tag = directive_match.groups()[0]
                sphinx_parts.append({
                    'tag': tag, 'start_offset': idx,
                    'type': 'directive',
                })

        prev_offset = len(lines)
        for part in sphinx_parts[::-1]:
            part['end_offset'] = prev_offset
            prev_offset = part['start_offset']

        for part in sphinx_parts[::-1]:
            if part['tag'] == 'returns':
                edit_slice = slice(part['start_offset'] + 2, part['end_offset'])
                return_section = lines[edit_slice]
                text = '\n'.join(return_section)

                new_lines = []
                for para in text.split('\n\n'):
                    indent = para[:len(para) - len(para.lstrip())]
                    new_paragraph = indent + paragraph(para)
                    new_lines.append(new_paragraph)
                    new_lines.append('')
                new_lines = new_lines[:-1]
                lines[edit_slice] = new_lines

        # print('AFTER:')
        # print('lines = {}'.format(ub.repr2(lines, nl=1)))

        # if name == 'kwimage.Affine.translate':
        #     import sys
        #     sys.exit(1)


def paragraph(text):
    r"""
    Wraps multi-line strings and restructures the text to remove all newlines,
    heading, trailing, and double spaces.

    Useful for writing log messages

    Args:
        text (str): typically a multiline string

    Returns:
        str: the reduced text block
    """
    import re
    out = re.sub(r'\s\s*', ' ', text).strip()
    return out


def create_doctest_figure(app, obj, name, lines):
    """
    The idea is that each doctest that produces a figure should generate that
    and then that figure should be part of the docs.
    """
    import xdoctest
    import sys
    import types
    if isinstance(obj, types.ModuleType):
        module = obj
    else:
        module = sys.modules[obj.__module__]
    if '--show' not in sys.argv:
        sys.argv.append('--show')
    if '--nointeract' not in sys.argv:
        sys.argv.append('--nointeract')
    modpath = module.__file__

    # print(doctest.format_src())
    import pathlib
    doc_outdir = pathlib.Path(app.outdir)
    # fig_dpath = (doc_outdir / 'autofigs' / name).mkdir(exist_ok=True)
    fig_dpath = (doc_outdir / '_static/images')
    fig_dpath.mkdir(exist_ok=True, parents=True)

    fig_num = 1

    import kwplot
    kwplot.autompl(force='agg')
    plt = kwplot.autoplt()

    docstr = '\n'.join(lines)

    # TODO: The freeform parser does not work correctly here.
    # We need to parse out the sphinx (epdoc)? individual examples
    # so we can get different figures. But we can hack it for now.

    import re
    split_parts = re.split('({}\\s*\n)'.format(re.escape('.. rubric:: Example')), docstr)
    # split_parts = docstr.split('.. rubric:: Example')

    # import xdev
    # xdev.embed()

    to_insert_fpaths = []

    def doctest_line_offsets(doctest):
        # Where the doctests starts and ends relative to the file
        start_line_offset = doctest.lineno - 1
        last_part = doctest._parts[-1]
        last_line_offset = start_line_offset + last_part.line_offset + last_part.n_lines - 1
        offsets = {
            'start': start_line_offset,
            'end': last_line_offset,
            'stop': last_line_offset + 1,
        }
        return offsets

    # from xdoctest import utils
    # part_lines = utils.add_line_numbers(docstr.split('\n'), n_digits=3, start=0)
    # print('\n'.join(part_lines))

    curr_line_offset = 0
    for part in split_parts:
        num_lines = part.count('\n')

        doctests = list(xdoctest.core.parse_docstr_examples(
            part, modpath=modpath, callname=name))
        # print(doctests)

        # doctests = list(xdoctest.core.parse_docstr_examples(
        #     docstr, modpath=modpath, callname=name))

        for doctest in doctests:
            if '--show' in part:
                ...
                # print('-- SHOW TEST---')/)
                # kwplot.close_figures()
                try:
                    import pytest  # NOQA
                except ImportError:
                    pass
                try:
                    from xdoctest.exceptions import Skipped
                except ImportError:  # nocover
                    # Define dummy skipped exception if pytest is not available
                    class Skipped(Exception):
                        pass
                try:
                    doctest.mode = 'native'
                    doctest.run(verbose=0, on_error='raise')
                    ...
                except Skipped:
                    print(f'Skip doctest={doctest}')
                except Exception as ex:
                    print(f'ex={ex}')
                    print(f'Error in doctest={doctest}')

                offsets = doctest_line_offsets(doctest)
                doctest_line_end = curr_line_offset + offsets['stop']
                insert_line_index = doctest_line_end

                figures = kwplot.all_figures()
                for fig in figures:
                    fig_num += 1
                    # path_name = path_sanatize(name)
                    path_name = (name).replace('.', '_')
                    fig_fpath = fig_dpath / f'fig_{path_name}_{fig_num:03d}.jpeg'
                    fig.savefig(fig_fpath)
                    print(f'Wrote figure: {fig_fpath}')
                    to_insert_fpaths.append({
                        'insert_line_index': insert_line_index,
                        'fpath': fig_fpath,
                    })

                for fig in figures:
                    plt.close(fig)
                # kwplot.close_figures(figures)

        curr_line_offset += (num_lines)

    # if len(doctests) > 1:
    #     doctests
    #     import xdev
    #     xdev.embed()

    INSERT_AT = 'end'
    INSERT_AT = 'inline'

    end_index = len(lines)
    # Reverse order for inserts
    for info in to_insert_fpaths[::-1]:
        rel_fpath = info['fpath'].relative_to(doc_outdir)

        if INSERT_AT == 'inline':
            # Try to insert after test
            insert_index = info['insert_line_index']
        elif INSERT_AT == 'end':
            insert_index = end_index
        else:
            raise KeyError(INSERT_AT)
        lines.insert(insert_index, '.. image:: {}'.format(rel_fpath))
        lines.insert(insert_index, '')

    # print('final lines = {}'.format(ub.repr2(lines, nl=1)))


def setup(app):
    import sphinx
    app : sphinx.application.Sphinx = app
    # sphinx.application.Sphinx
    app.add_domain(PatchedPythonDomain, override=True)
    docstring_processor = GoogleStyleDocstringProcessor()
    if 1:
        # New Way
        # what = None
        app.connect('autodoc-process-docstring', docstring_processor.process_docstring_callback)
    else:
        # OLD WAY
        # https://stackoverflow.com/questions/26534184/can-sphinx-ignore-certain-tags-in-python-docstrings
        # Register a sphinx.ext.autodoc.between listener to ignore everything
        # between lines that contain the word IGNORE
        # from sphinx.ext.autodoc import between
        # app.connect('autodoc-process-docstring', between('^ *Ignore:$', exclude=True))
        pass
    return app
