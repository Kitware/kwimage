"""
https://github.com/martin-steinegger/setcover

g++ -o setcover_cpp.so -Wall -m64 -ffast-math -ftree-vectorize -O3 -Wno-write-strings SetCover.cpp setcover.cpp
"""


def define_ext():
    from distutils.extension import Extension
    ext = Extension(
        name='setcover_cyth',
        sources=['setcover_cyth.pyx', 'SetCover.cpp'],
        # extra_compile_args=["-std=c++11 -Wall -m64 -ffast-math -ftree-vectorize -O3 -Wno-write-strings"],
        language="c++",             # generate C++ code
    )
    from Cython.Build import cythonize
    ext_modules = cythonize(ext)  # this just transforms to a cpp file
    return ext_modules


def dobuild():
    ext_modules = define_ext()
    build_extension(ext_modules[0])


def build_extension(ext):
    """
    HUGE hack around setuptools / distutils to programatically build a cython
    extension.

    Based on simplified pyxbuild code:
        https://github.com/cython/cython/blob/master/pyximport/pyxbuild.py

    DOESNT WORK FOR SOME REASON
    """
    from distutils.dist import Distribution
    from Cython.Distutils import build_ext
    sargs = {
        "script_name": None,
        "script_args": ["build_ext"],
    }
    dist = Distribution(sargs)
    dist.ext_modules = [ext]
    dist.cmdclass = {'build_ext': build_ext}

    build = dist.get_command_obj('build')
    import os
    pyxbuild_dir = os.path.join('.', "_pyxbld")
    build.build_base = pyxbuild_dir

    cfgfiles = dist.find_config_files()
    dist.parse_config_files(cfgfiles)

    ok = dist.parse_command_line()
    assert ok
    import sys
    try:
        obj_build_ext = dist.get_command_obj("build_ext")
        dist.run_commands()
        so_path = obj_build_ext.get_outputs()[0]
        if obj_build_ext.inplace:
            # Python distutils get_outputs()[ returns a wrong so_path
            # when --inplace ; see http://bugs.python.org/issue5977
            # workaround:
            so_path = os.path.join('.', os.path.basename(so_path))
    except KeyboardInterrupt:
        sys.exit(1)
    except (IOError, os.error):
        raise
    assert ok
    return so_path


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwil/kwil/util/setcover/_build.py
        python _build.py build_ext --inplace
    """
    from distutils.core import setup
    from Cython.Distutils import build_ext
    ext_modules = define_ext()
    setup(ext_modules=ext_modules, cmdclass={'build_ext': build_ext})
