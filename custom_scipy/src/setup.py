from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("custom_scipy/src/peak_finding_utils.pyx"))
