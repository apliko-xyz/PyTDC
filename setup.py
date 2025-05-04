from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

# get __version__ from _version.py
ver_file = path.join("tdc_ml", "version.py")
with open(ver_file) as f:
    exec(f.read())

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.md
def readme():
    with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
        return f.read()


# read the contents of requirements.txt
with open(path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="pytdc-nextml",
    version=__version__,
    license="MIT",
    description=
    "PyTDC: A multimodal machine learning training, evaluation, and inference platform for biomedical foundation models",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/apliko-xyz/PyTDC",
    author="PyTDC Team",
    author_email="amva13@alum.mit.edu",
    packages=find_packages(exclude=["test"]),
    zip_safe=False,
    include_package_data=True,
    install_requires=requirements,
    setup_requires=["setuptools>=38.6.0"],
    python_requires=">=3.9",
)
