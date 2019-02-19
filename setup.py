#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import exists
from os.path import join
from skbuild import setup
from setuptools import find_packages
import sys
from os.path import dirname


repodir = dirname(__file__)


def parse_version(package):
    """
    Statically parse the version number from __init__.py

    CommandLine:
        python -c "import setup; print(setup.parse_version('kwimage'))"
    """
    from os.path import dirname, join, exists
    import ast

    # Check if the package is a single-file or multi-file package
    _candiates = [
        join(dirname(__file__), package + '.py'),
        join(dirname(__file__), package, '__init__.py'),
    ]
    _found = [init_fpath for init_fpath in _candiates if exists(init_fpath)]
    if len(_found) > 0:
        init_fpath = _found[0]
    elif len(_found) > 1:
        raise Exception('parse_version found multiple init files')
    elif len(_found) == 0:
        raise Exception('Cannot find package init file')

    with open(init_fpath) as file_:
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


def parse_description():
    """
    Parse the description in the README file

    CommandLine:
        pandoc --from=markdown --to=rst --output=README.rst README.md
        python -c "import setup; print(setup.parse_description())"
    """
    from os.path import dirname, join, exists
    readme_fpath = join(dirname(__file__), 'README.rst')
    # This breaks on pip install, so check that it exists.
    if exists(readme_fpath):
        with open(readme_fpath, 'r') as f:
            text = f.read()
        return text
    return ''


def parse_requirements_alt(fname='requirements.txt'):
    """
    pip install requirements-parser
    fname='requirements.txt'
    """
    import requirements
    from os.path import dirname, join, exists
    require_fpath = join(dirname(__file__), fname)
    if exists(require_fpath):
        # Dont use until this handles platform specific dependencies
        with open(require_fpath, 'r') as file:
            requires = list(requirements.parse(file))
        packages = [r.name for r in requires]
        return packages
    return []


def parse_requirements(fname='requirements.txt'):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    TODO:
        perhaps use https://github.com/davidfischer/requirements-parser instead

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    from os.path import dirname, join, exists
    import re
    require_fpath = join(dirname(__file__), fname)

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        info = {}
        if line.startswith('-e '):
            info['package'] = line.split('#egg=')[1]
        else:
            # Remove versioning from the package
            pat = '(' + '|'.join(['>=', '==', '>']) + ')'
            parts = re.split(pat, line, maxsplit=1)
            parts = [p.strip() for p in parts]

            info['package'] = parts[0]
            if len(parts) > 1:
                op, rest = parts[1:]
                if ';' in rest:
                    # Handle platform specific dependencies
                    # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                    version, platform_deps = map(str.strip, rest.split(';'))
                    info['platform_deps'] = platform_deps
                else:
                    version = rest  # NOQA
                info['version'] = (op, version)
        return info

    # This breaks on pip install, so check that it exists.
    if exists(require_fpath):
        with open(require_fpath, 'r') as f:
            packages = []
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    info = parse_line(line)
                    package = info['package']
                    if not sys.version.startswith('3.4'):
                        # apparently package_deps are broken in 3.4
                        platform_deps = info.get('platform_deps')
                        if platform_deps is not None:
                            package += ';' + platform_deps
                    packages.append(package)
            return packages
    return []


def clean():
    """
    __file__ = ub.truepath('~/code/kwimage/setup.py')
    """
    import ubelt as ub
    import os
    import glob

    modname = 'kwimage'
    repodir = dirname(os.path.realpath(__file__))

    toremove = []
    for root, dnames, fnames in os.walk(repodir):

        if os.path.basename(root) == modname + '.egg-info':
            toremove.append(root)
            del dnames[:]

        if os.path.basename(root) == '__pycache__':
            toremove.append(root)
            del dnames[:]

        if os.path.basename(root) == '_ext':
            # Remove torch extensions
            toremove.append(root)
            del dnames[:]

        if os.path.basename(root) == 'build':
            # Remove python c extensions
            if len(dnames) == 1 and dnames[0].startswith('temp.'):
                toremove.append(root)
                del dnames[:]

        # Remove simple pyx inplace extensions
        for fname in fnames:
            if fname.endswith(('.so', '.c', '.o')):
                if fname.split('.')[0] + '.pyx' in fnames:
                    toremove.append(join(root, fname))

    def enqueue(d):
        if exists(d) and d not in toremove:
            toremove.append(d)

    enqueue(join(repodir, 'htmlcov'))
    enqueue(join(repodir, 'kwimage/algo/_nms_backend/cpu_nms.c'))
    enqueue(join(repodir, 'kwimage/algo/_nms_backend/cpu_nms.cpp'))
    enqueue(join(repodir, 'kwimage/algo/_nms_backend/gpu_nms.cpp'))
    enqueue(join(repodir, 'kwimage/structs/_boxes_backend/cython_boxes.c'))
    enqueue(join(repodir, 'kwimage/structs/_boxes_backend/cython_boxes.html'))
    for d in glob.glob(join(repodir, 'kwimage/algo/_nms_backend/*_nms.*so')):
        enqueue(d)

    for d in glob.glob(join(repodir, 'kwimage/structs/_boxes_backend/cython_boxes*.*so')):
        enqueue(d)

    enqueue(join(repodir, '_skbuild'))
    enqueue(join(repodir, '_cmake_test_compile'))
    enqueue(join(repodir, 'kwimage.egg-info'))
    enqueue(join(repodir, 'pip-wheel-metadata'))

    for dpath in toremove:
        ub.delete(dpath, verbose=1)


# Scikit-build extension module logic
compile_setup_kw = dict(
    cmake_languages=('C', 'CXX', 'CUDA'),
    cmake_source_dir='.',
    # cmake_source_dir='kwimage',
)

try:
    import numpy as np
    # Note: without this skbuild will fail with `pip install -e .`
    # however, it will still work with `./setup.py develop`.
    # Not sure why this is, could it be an skbuild bug?
    compile_setup_kw['cmake_args'] = [
        '-DNumPy_INCLUDE_DIR:PATH=' + np.get_include()
    ]
except ImportError:
    pass


version = parse_version('kwimage')  # needs to be a global var for git tags

if __name__ == '__main__':
    if 'clean' in sys.argv:
        # hack
        clean()
        sys.exit(0)
    setup(
        name='kwimage',
        version=version,
        author='Jon Crall',
        long_description=parse_description(),
        install_requires=parse_requirements('requirements.txt'),
        author_email='erotemic@gmail.com',
        url='https://kwgitlab.kitware.com/jon.crall/kwimage',
        packages=find_packages(include='kwimage.*'),
        classifiers=[
            # List of classifiers available at:
            # https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 3 - Alpha',
            # This should be interpreted as Apache License v2.0
            'License :: OSI Approved :: Apache Software License',
            # Supported Python versions
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        **compile_setup_kw,
    )
