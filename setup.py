"""
Setup configuration for PDF Q&A System.
"""
from setuptools import setup, find_packages

setup(
    name="pdf-qa-app-with-langchain",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.8",
) 