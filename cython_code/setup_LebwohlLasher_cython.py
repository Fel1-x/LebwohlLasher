from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="LebwohlLasher_cython",
      include_dirs=[numpy.get_include()],
      ext_modules=cythonize("LebwohlLasher_cython.pyx")
)

