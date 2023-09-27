#!/usr/bin/env python

import re
import io
import os
import shutil
import distutils.command.clean
from pathlib import Path

# deprecated, but ML directory layout departs from current recommendations
from setuptools import setup, find_namespace_packages
# N/A for aarch64:
#from pkg_resources import get_distribution, DistributionNotFound

from ml.shutil import run as sh

cwd = Path(__file__).parent
pkg = sh('basename -s .git `git config --get remote.origin.url`').lower()
PKG = pkg.upper()

import platform
host_arch = platform.machine() # x86_64 or aarch64?

def write_version_py(path, major=None, minor=None, patch=None, suffix='', sha='Unknown'):
    if major is None or minor is None or patch is None:
        major, minor, patch = sh("git tag --sort=taggerdate | tail -1")[1:].split('.')
        sha = sh("git rev-parse HEAD")
        print(f"Build version {major}.{minor}.{patch}-{sha}")

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
import ml, torch
__version__ = {repr(version)}
git_version = {repr(sha)}
cuda = torch.version.cuda
"""

    with open(path / 'version.py', 'w') as f:
        f.write(content)
    
    return version

def write_test_ml_imports_py(packages):
    import time
    header = f"""# GENERATED TEST FILE
# TIME: {time.asctime()}"
"""
    imports = '\n'.join([f"print('import {p} ...')\nimport {p}" for p in packages])
    with open("./test_ml_imports.py", "w") as f:
        f.write(header)
        f.write(imports)

    return None

#def dist_info(pkgname):
#    try:
#        return get_distribution(pkgname)
#    except DistributionNotFound:
#        return None


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


#import packaging.version
#torch_ver = packaging.version.parse(torch.__version__)

if __name__ == '__main__':
    version = write_version_py(pkg)
    if host_arch == 'aarch64':
        # The makefile pre-installs and pins the pytorch version for JetPack
        requirements = [
                # for JetPack 5.1.2, pytorch 2.1.0_cp38; or pytorch 2.0.0_cp39
                # prebuilt wheels as per https://forums.developer.nvidia.com
                #    Following suggestion had syntax error:
                #pip @ file:///mnt/jetson/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
                'torch>=2.0'
                ]
    else:
        requirements = [
                'torch',
                'numexpr',
                'pyyaml',
                'requests',
                'requests-toolbelt',
                'av',
                ]

    namespaces = ['ml']
    # following is DEPRECATED:
    packages = find_namespace_packages(include=['ml.*'], exclude=('ml.csrc', 'ml.csrc.*'))
    unsupported = []
    if host_arch == 'aarch64': # and torch_ver >= packaging.version.parse("2.0"):
        # ml.tasks is based upon pytorch/ignite, which may have issues with 2.0+
        unsupported.append('ml.tasks')
        unsupported.append('ml.tasks.detection') # requires pytorch/ignite, which may have issues wtih 2.0+
        unsupported.append('ml.tasks.grounding') # requires pytorch/ignite, which may have issues wtih 2.0+
        unsupported.append('ml.tasks.detection.coco') # requires pytorch/ignite, which may have issues wtih 2.0+
        # ml.av also failed to 'import' ??

    packages = [p for p in packages if p not in unsupported]

    packages = namespaces + packages 
    print("setup.py packages: ",packages)
    write_test_ml_imports_py(packages)

    setup(
        name=pkg.upper(),
        version=version,
        author='Farley Lai;Deep Patel',
        url='https://gitlab.com/necla-ml/ML',
        description=f"NECLA ML Library",
        long_description=readme(),
        keywords='machine learning, computer vision, natural language processing, distributed computing',
        license='BSD-3',
        classifiers=[
            'License :: OSI Approved :: BSD License',
            'Operating System :: macOS/Ubuntu 18.04+',
            'Development Status :: 1 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        packages=namespaces + packages,
        install_requires=requirements,
        zip_safe=False)
