import os
from setuptools import setup

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        fcontent = f.read()
    return fcontent
    

setup(
    name = "explainability_calibration",
    version = "0.0.1",
    author = "Lautaro Estienne",
    author_email = "lestienne@fi.uba.ar",
    description = ("Code to work on the interpretabiliy of the model and its calibration"),
    keywords = "explainability calibration",
    url = "https://github.com/LautaroEst/explainability-calibration",
    packages=['explainability_calibration'],
    long_description=read('Readme.md')
)