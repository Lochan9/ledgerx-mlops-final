from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ledgerx",
    version="1.0.0",
    author="LedgerX Team",
    author_email="team@ledgerx.com",
    description="Enterprise Invoice Intelligence Platform with MLOps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lochan9/ledgerx-mlops-final",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "catboost>=1.2.0",
        "mlflow>=2.12.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "pydantic>=2.8.0",
        "python-jose>=3.3.0",
        "passlib>=1.7.0",
        "bcrypt>=4.1.0",
        "google-cloud-documentai>=2.20.0",
        "google-cloud-storage>=2.10.0",
        "prometheus-client>=0.19.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
)