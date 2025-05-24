# setup.py
from setuptools import setup, find_packages

setup(
    name="constraintflow",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "antlr4-python3-runtime==4.7.2",
        "tabulate",
    ],
    python_requires='>=3.7',
    include_package_data=True,
)
