#!/usr/bin/env python
from numpy.distutils.core import setup, Extension
import sys

if len(sys.argv)<2:
    print("see install.txt for installation instructions.")



instpackage = ["pyasf"]
try:
    import CifFile
except:
    instpackage.append("CifFile")

setup( name = "pyasf", 
       version = "0.1",
       packages = instpackage,
       package_data = {"pyasf": ["space-groups.sqlite", "settings.txt", "f0_lowq.sqlite"]},
       author = "Carsten Richter",
       author_email = "carsten.richter@desy.de",
       description = "Software for symbolical calculation of the anisotropic tensor of susceptibility and the anisotropic structure factor (ASF).",
       long_description = """
                             Module provides symbolical calculation of the anisotropic Structure Factor
                             of a Crystal of given Space Group and Asymmetric Unit up todipole-quadrupole (DQ) approximation.
                             This class represents the unit cell of a crystal in terms of Resonant
                             Elastic X-Ray Scattering (REXS)
                          """
     )

