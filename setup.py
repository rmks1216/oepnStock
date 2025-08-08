"""
Setup script for oepnStock - Korean Stock Market Trading System
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="oepnstock",
    version="1.0.0",
    author="Claude Code",
    author_email="claude@anthropic.com",
    description="Korean Stock Market Trading System with 4-Stage Checklist Strategy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/oepnstock",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "oepnstock=oepnstock.cli:main",
        ],
    },
    package_data={
        "oepnstock": [
            "config/*.yaml",
            "config/*.yml", 
            "data/*.csv",
            "data/*.json",
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords="stock trading korean market algorithmic trading KOSPI KOSDAQ",
    project_urls={
        "Documentation": "https://oepnstock.readthedocs.io/",
        "Source": "https://github.com/your-username/oepnstock",
        "Tracker": "https://github.com/your-username/oepnstock/issues",
    },
)