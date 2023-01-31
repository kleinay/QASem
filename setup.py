import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("qasem/version.txt", "r") as f:
    version = f.read().strip()

setuptools.setup(
    name="qasem",
    version=version,
    author="Ayal Klein, Ruben Wolhandler, Ron Eliav",
    author_email="ayal.s.klein@gmail.com",
    description="package for QA-based Semantics - representing textual information via question-answer pairs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kleinay/QASem",
    packages=setuptools.find_packages(),
    install_requires=[
        'transformers==4.15.0',
        'spacy>=3.0.0',
        'qanom>=0.0.30',
        'constrained_decoding',
        'markupsafe==2.0.1', # downgrade because of bug in 2.1.1, https://stackoverflow.com/questions/72191560/importerror-cannot-import-name-soft-unicode-from-markupsafe
    ],
    package_data={
        "": ["qasem/data/connectives_small_set.txt"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
