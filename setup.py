from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="delayed-coking-cfd",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CFD simulation of delayed coking reactor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/delayed-coking-cfd",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
        "pandas>=1.3.0",
        "numba>=0.54.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "pytest-cov>=2.12.0", "black", "flake8"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
)