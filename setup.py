#!/usr/bin/env python

import re
import io
import os
import sys
import glob
import shutil
import subprocess
import distutils.command.clean
from pathlib import Path

from setuptools import setup, find_packages
from pkg_resources import get_distribution, DistributionNotFound
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

from ml.shutil import run as sh
from ml import logging

'''
MAJOR = 0
MINOR = 1
PATCH = 1
SUFFIX = ''
'''

cwd = Path(__file__).parent
pkg = sh('basename -s .git `git config --get remote.origin.url`').lower()
PKG = pkg.upper()

def write_version_py(path, major=None, minor=None, patch=None, suffix='', sha='Unknown'):
    if major is None or minor is None or patch is None:
        major, minor, patch = sh("git describe --abbrev=0 --tags")[1:].split('.')
        sha = sh("git rev-parse HEAD")
        logging.info(f"Build version {major}.{minor}.{patch}-{sha}")

    path = Path(path).resolve()
    pkg = path.name
    PKG = pkg.upper()
    version = f'{major}.{minor}.{patch}{suffix}'
    if os.getenv(f'{PKG}_BUILD_VERSION'):
        assert os.getenv(f'{PKG}_BUILD_NUMBER') is not None
        build_number = int(os.getenv(f'{PKG}_BUILD_NUMBER'))
        version = os.getenv(f'{PKG}_BUILD_VERSION')
        if build_number > 1:
            version += '.post' + str(build_number)
    elif sha != 'Unknown':
        version += '+' + sha[:7]

    import time
    content = f"""# GENERATED VERSION FILE
# TIME: {time.asctime()}
__version__ = {repr(version)}
git_version = {repr(sha)}

from ml import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
"""

    with open(path / 'version.py', 'w') as f:
        f.write(content)
    
    return version


def dist_info(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


def ext_modules(pkg):
    pkg_dir = cwd / pkg
    extensions_dir = pkg_dir / 'csrc'

    main_files = sorted(map(str, extensions_dir.glob('*.cpp')))
    source_cpu = sorted(map(str, (extensions_dir / 'cpu').glob('*.cpp')))
    source_cuda = sorted(map(str, (extensions_dir / 'cuda').glob('*.cu')))

    sources = main_files + source_cpu
    extension = CppExtension
    if not sources:
        return []

    test_dir = cwd / 'tests'
    test_files = sorted(map(str, test_dir.glob('*.cpp')))
    tests = test_files

    define_macros = []
    extra_compile_args = {}
    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv('FORCE_CUDA', '0') == '1':
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        if nvcc_flags == '':
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(' ')
        extra_compile_args = {
            'cxx': ['-O3'],
            'nvcc': nvcc_flags,
        }

    if sys.platform == 'win32':
        define_macros += [('{pkg}_EXPORTS', None)]

    include_dirs = [str(extensions_dir)]
    tests_include_dirs = [str(test_dir)]
    ext_modules = [
        extension(
            f'{pkg}._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )]
    if tests:
        ext_modules.append(extension(
            f'{pkg}._C_tests',
            tests,
            include_dirs=tests_include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ))
    return ext_modules

# TODO
class Clean(distutils.command.clean.clean):
    def run(self):
        import glob
        import re
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        print(f"removing {filename} to clean")
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


if __name__ == '__main__':
    version = write_version_py(pkg)
    requirements = [
        'torch',
        'torchvision',
#        'torchaudio',
#        'torchtext',
#        'transformers'
    ]
    cmdclass = dict(
        build_ext=torch.utils.cpp_extension.BuildExtension,
        clean=Clean,
    )
    extensions = [ext for ext in ext_modules(pkg)]
    packages = find_packages(exclude=('tools', 'tools.*', 'recipe', 'submodules'))
    setup(
        name=pkg.upper(),
        version=version,
        author='Farley Lai',
        url='https://gitlab.com/necla-ml/ML',
        description=f"NECLA ML Library",
        long_description=readme(),
        keywords='machine learning, computer vision, natural language processing, distributed computing',
        license='BSD-3',
        classifiers=[
            'License :: OSI Approved :: BSD License',
            'Operating System :: macOS/Ubuntu 16.04+',
            'Development Status :: 1 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        packages=packages,
        # package_data={f'{pkg}.ops': ['*/*.so']},
        # setup_requires=['pytest-runner'],
        # tests_require=['pytest'],
        install_requires=requirements,
        extras_require={
            #"scipy": ["scipy"],
        },
        ext_modules=extensions,
        cmdclass=cmdclass,
        zip_safe=False)
