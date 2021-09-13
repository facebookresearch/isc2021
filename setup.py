#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import setuptools
from pathlib import Path


requirements = [
    r
    for r in Path("requirements.txt").read_text().splitlines()
    if '@' not in r
]


setuptools.setup(
    name="isc2021",
    version="0.0.1",
    description="Code for the image similarity challenge at NeurIPS 2021",
    url="https://github.com/facebookresearch/isc2021",
    author="Matthijs Douze",
    author_email="matthijs@fb.com",
    packages=["isc"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.5",
)

