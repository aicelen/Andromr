from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(r"/home/enno/andromr/custom_skimage/src/_warps_cy.pyx")
)
