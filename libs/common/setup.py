from setuptools import setup, find_packages

setup(
    name="common",
    version="0.0.1",
    author="Daniel De Lucca Fonseca",
    author_email="daniel@delucca.dev",
    description="Common libraries and tools for the Memory Aware Chunking experiments",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/delucca-papers/memory-aware-chunking",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
