from setuptools import setup, find_packages

setup(
    name="latent_flow",
    version="0.1.0",
    author="Yue Song",
    python_requires=">=3.8.0",
    packages=find_packages(exclude=("tests", "docs", "imgs")),
    url="https://github.com/KingJamesSong/latent-flow",
    description='NeurIPS23 "Flow Factorized Representation Learning"',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
