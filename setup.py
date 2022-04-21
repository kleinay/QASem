import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

version = "0.0.1"

setuptools.setup(
    name="qasem",
    version=version,
    author="Ayal Klein, Ruben Wolhandler",
    author_email="ayal.s.klein@gmail.com",
    description="package for QA-based Semantics - representing textual information via question-answer pairs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kleinay/QASem",
    packages=setuptools.find_packages(),
    install_requires=[
        'transformers==4.15.0',
        'spacy==2.3.7',
        'qanom',
        'roleqgen @ git+https://github.com/rubenwol/RoleQGeneration.git@main#egg=RoleQGeneration'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)