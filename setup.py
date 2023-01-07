import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "leap_sd",
    version = "0.0.1",
    author = "Peter Willemsen",
    author_email = "peter@codebuffet.co",
    description = "Fast finetuning of Stable Diffusion using LEAP booster model.",
    license = "MIT",
    keywords = "finetuning training stable-diffusion huggingface",
    url = "https://github.com/peterwilli/sd-leap-booster",
    packages=['leap_sd'],
    long_description=read('README.md'),
    scripts=['bin/leap_textual_inversion'],
    install_requires=[
        'numpy',
        'diffusers',
        'transformers',
        'datasets',
        'torchvision',
        'accelerate',
        'pytorch_lightning',
        'tensorboard'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
) 
