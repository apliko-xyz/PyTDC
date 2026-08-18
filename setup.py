from io import open  # for Python 2 and 3 compatibility
from os import path

from setuptools import find_packages, setup

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
    license_files=("LICENSE",),
    description=
    "PyTDC: A multimodal machine learning training, evaluation, and inference platform for biomedical foundation models",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/apliko-xyz/PyTDC",
    project_urls={
        "Homepage": "https://pytdc.apliko.io",
        "Source": "https://github.com/apliko-xyz/PyTDC",
        "Issues": "https://github.com/apliko-xyz/PyTDC/issues",
    },
    author="PyTDC Team",
    author_email="amva13@alum.mit.edu",
    packages=find_packages(exclude=["test"]),
    zip_safe=False,
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.9,<3.15",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
