try:
    from setuptools import setup
except ImportError:
    from distutils import setup
from distutils.extension import Extension

import numpy as np
# setuptools DWIM monkey-patch madness
# http://mail.python.org/pipermail/distutils-sig/2007-September/thread.html#8204
import sys

if 'setuptools.extension' in sys.modules:
    m = sys.modules['setuptools.extension']
    m.Extension.__dict__ = m._Extension.__dict__
gsw_extension = Extension("pygsw.seawater", ["pygsw/seawater.pyx"], include_dirs=[np.get_include()], libraries=['gswteos-10'] )


classifiers = ''' Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Education
Topic :: Software Development :: Libraries :: Python Modules'''
setup(name = 'pygsw', 
        version='0.0.1',
        description='Python bindings for the TEOS-10 V3.0 GSW Oceanographic Toolbox in C',
        long_description=open('README.md').read(),
        license='LICENSE.txt',
        author='Luke Campbell',
        author_email='luke.s.campbell@gmail.com',
        url='https://github.com/lukecampbell/pygsw/',
        ext_modules=[gsw_extension],
        classifiers=classifiers.split('\n'),
        packages=['pygsw'],
        keywords=['oceanography', 'seawater'],
        setup_requires=['setuptools_cython'],
        )


