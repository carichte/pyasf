#!/usr/bin/env python
import os
import subprocess
import sys
from setuptools import setup
from setuptools import find_packages


if len(sys.argv)<2:
    print("see install.txt for installation instructions.")



setup( name = "pyasf", 
       version = "0.1",
       packages = find_packages(),
       package_data = {
            "pyasf": ["space-groups.sqlite", 
                      "settings.txt.gz", 
                      "f0_lowq.sqlite",
                      "coppens/*"],
            "pyasf.materials": ["cif/*"]},
       author = "Carsten Richter",
       author_email = "carsten.richter@desy.de",
       description = "Software for symbolical calculation of the anisotropic tensor of susceptibility and the anisotropic structure factor (ASF).",
       long_description = """
                             Module provides symbolical calculation of the anisotropic Structure Factor
                             of a Crystal of given Space Group and Asymmetric Unit up to dipole-quadrupole (DQ) approximation.
                             This class represents the unit cell of a crystal in terms of Resonant
                             Elastic X-Ray Scattering (REXS)
                          """,
        install_requires=[
                      'numpy',
                      'sympy',
                      'PyCifRW',
                      'lxml'
                     ],
     )

