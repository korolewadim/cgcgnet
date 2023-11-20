# CGCGNet version 0.1

import setuptools


with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="cgcgnet",
    version="0.1",
    description="cgcgnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vadim Korolev, Artem Mitrofanov",
    author_email="korolewadim@gmail.com",
    include_package_data=True,
    package_data={"": ["models/*.pkl"]},
    install_requires=[
        "torch",
        "dgl",
        "matminer",
        "rdkit",
        "gensim",
    ],
    packages=setuptools.find_packages(),
)
