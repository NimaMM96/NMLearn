import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nm-learn",
    version="v0.0.1",
    author="Nima Mohammadi Meshky",
    author_email="nmmohammadi96@gmail.com",
    description="A small maching-learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NimaMM96/NMLearn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
