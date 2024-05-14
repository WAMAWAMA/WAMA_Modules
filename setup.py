from setuptools import find_packages, setup
import io
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except:
    REQUIRED = []


setup(
    name='aini_modules',
    version='0.0.1',
    description='Enjoy~',
    author='wamawama',
    author_email='wmy19970215@gmail.com',
    python_requires=">=3.6.0",
    url='https://github.com/WAMAWAMA/wama_modules',
    packages=find_packages(exclude=("demo", "docs", "images")),
    install_requires=REQUIRED,
    license="MIT",
)

