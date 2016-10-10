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
from materials import STO

NPx = 2048
PxSize = 45e-6
cenx, ceny = 1017, 1059
SDD = 400
ttoff = 20

ttoff = pl.radians(ttoff)
X, Y = pl.meshgrid(pl.arange(NPx), pl.arange(NPx)[::-1])
Xm = (X - cenx) * PxSize
Ym = (Y - ceny) * PxSize * pl.cos(ttoff) + pl.sin(ttoff)*SDD

theta = ################

cs = STO.get_cs()

theta = pl.linspace(5,80,10001)

I = cs.XRD_pattern(theta, 10000.)

pl.plot(theta, I)

pl.xlabel("theta (deg)")
pl.ylabel("intensity")
pl.show()




