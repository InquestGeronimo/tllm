from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    LONG_DESCRIPTION = "\n" + fh.read()

NAME = "cyphertune"
VERSION = "0.0.0.3"
AUTHOR = "InquestGeronimo"
EMAIL = "rcostanl@gmail.com"
LD_CONTENT_TYPE = "text/markdown"
DESCRIPTION = "A Trainer for Fine-tuning LLMs for Text-to-Cypher Datasets"
LICENSE = "Apache 2.0 license"
PACKAGES = find_packages()
DEPENDENCIES = [
    "accelerate>=0.27.0",
    "bitsandbytes>=0.42.0",
    "datasets>=2.17.0",
    "peft>=0.8.2",
    "transformers>=4.37.2",
    "wandb>=0.16.3",
    "scipy>=1.11.4",
    "pydantic>=2.6.1"
]
KEYWORDS = ["llms", "training", "fine-tuning", "LLM", "NLP"]
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Operating System :: Unix",
]

setup(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description_content_type=LD_CONTENT_TYPE,
    long_description=LONG_DESCRIPTION,
    packages=PACKAGES,
    include_package_data=True,
    install_requires=DEPENDENCIES,
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
)
