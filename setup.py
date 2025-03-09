import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from naml import __version__

setuptools.setup(
    name="naml",
    version=__version__,
    author="mos9527",
    author_email="greats3an@gmail.com",
    description="Naml is Another Machine Learning library (better name pending)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mos9527/naml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "matplotlib==3.10.1",
        "numpy==2.2.3",
        "scipy==1.15.2",
        "sympy==1.13.1",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        "nbformat",
        "tqdm",
        "requests",
        "pytest",
    ],
    python_requires="~=3.10",
)
