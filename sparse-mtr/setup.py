from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy


def readme():
    with open('README.md') as f:
        return f.read()

extensions = [
    Extension(
        "smtr.estimators.solvers.solver_cd",
        ['smtr/estimators/solvers/solver_cd.pyx'],
    ),
    Extension(
        "smtr.estimators.solvers.solver_cd_reweighted",
        ['smtr/estimators/solvers/solver_cd_reweighted.pyx'],
    ),
]

if __name__ == "__main__":
    setup(name="smtr",
          packages=find_packages(),
          ext_modules=cythonize(extensions),
          include_dirs=[numpy.get_include()]
          )
