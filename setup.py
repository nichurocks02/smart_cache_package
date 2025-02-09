from setuptools import setup, find_packages

setup(
    name="smart_cache_package",
    version="0.1.5",
    author="Nishkal Gupta M",
    author_email="nishkalrocks02@gmail.com",
    description="An intelligent caching and LLM routing package using Mem0AI and zero-shot classification.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nichurocks02/smart_cache_package.git",
    packages=find_packages(),
    install_requires=[
        "mem0AI", 
        "transformers",
        "openai",
        "dateparser",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.10',
)

