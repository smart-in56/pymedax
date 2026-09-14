from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


ext_modules = [
    Pybind11Extension(
        "pymedax._medial_axis",
        ["src/pymedax/_medial_axis.cpp"],
        cxx_std=17,
    ),
]

setup(ext_modules=ext_modules)
