#!/usr/bin/env python
#----------------------------------------------------------------------
# Description:
# Author: Carsten Richter <carsten.richter@desy.de>
# Created at: Do 13. Aug 18:12:01 CEST 2015
# Computer: haso227r 
# System: Linux 3.13.0-61-generic on x86_64
#
# Copyright (c) 2015 Carsten Richter  All rights reserved.
#----------------------------------------------------------------------
import os
import pylab as pl
#from rexs import tools
#import pyasf


#cs = pyasf.unit_cell("9006864")
from pyasf.materials import STO

cs = STO.get_cs()
theta = pl.linspace(5,80,10001)

I = cs.XRD_pattern(theta, 10000.)

pl.plot(theta, I)

pl.xlabel("theta (deg)")
pl.ylabel("intensity")
pl.show()




