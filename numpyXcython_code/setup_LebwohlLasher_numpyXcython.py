from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="LebwohlLasher_numpyXcython",
      include_dirs=[numpy.get_include()],
      ext_modules=cythonize("LebwohlLasher_numpyXcython.pyx")
)

