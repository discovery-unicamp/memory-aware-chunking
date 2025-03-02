from setuptools import setup, find_packages

setup(
    name="traceq",
    version="0.0.1",
    author="Daniel De Lucca Fonseca",
    author_email="daniel@delucca.dev",
    description="A precise profiler for Python, optimized for experiment processing tasks in high-performance computing. Capable of sampling with metadata, using minimal instrumentation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/discovery-unicamp/traceq",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "traceq=traceq.cli:cli",
        ],
    },
)
