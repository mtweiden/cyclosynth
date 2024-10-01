"""Set up file for cyclosynth."""
from setuptools import setup, find_packages

setup(
    name='cyclosynth',
    description='For compiling to Fault-Tolerant cyclotomic gate sets.',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['sympy'],
    python_requires='>=3.8, <4.0'
)
