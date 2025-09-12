# setup.py
from setuptools import find_packages, setup

setup(
    name="constraintflow",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "antlr4-python3-runtime==4.13.2",
        "tabulate",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
