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
setup(name = 'pygsw', 
        vesrion='0.1',
        description='',
        ext_modules=[gsw_extension],
        packages=['pygsw'],
        setup_requires=['setuptools_cython'],
        )


