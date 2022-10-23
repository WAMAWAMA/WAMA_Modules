from setuptools import find_packages, setup

setup(
    name='wama_modules',
    version='1.0.0',
    description='Nothing',
    author='wamawama',
    author_email='wmy19970215@gmail.com',
    python_requires=">=3.6.0",
    url='https://github.com/WAMAWAMA/wama_modules',
    # packages=find_packages(exclude=("tests", "docs", "images")),
    packages=find_packages(),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=[],
    license="MIT",
)