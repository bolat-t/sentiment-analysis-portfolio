from setuptools import setup, find_packages

setup(
    name="sentiment-analysis-portfolio",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive sentiment analysis toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "transformers>=4.21.0",
        "torch>=1.12.0",
    ],
)
