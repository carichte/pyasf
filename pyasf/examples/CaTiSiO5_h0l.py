#!/usr/bin/env python
"""
Outer Batch file for pyasf.py.

Written by Carsten Richter (carsten.richter@desy.de)

"""
import materials.HPS as hps
import itertools
import pylab as pl
import pyasf
import time
mydict = dict({"Abs":abs})


cs = pyasf.unit_cell("MyBaseFileName_9837.cif")


hmax = 20
lmax = 20
energy = 10000.
hindizes = range( -hmax, hmax+1)
lindizes = range( -lmax, lmax+1)

F = cs.get_F0(energy = energy, equivalent=True)
h, k, l = cs.S["h"], cs.S["k"], cs.S["l"]
F = F.subs(k,0)

ha = pl.linspace(-2, 2, 801)*10
la = pl.linspace(-2, 2, 801)*10

Imap = 0 * ha * la[:,pl.newaxis]
I0 = float(abs(F.subs({h:0, l:0}).n())**2)

for miller in itertools.product(hindizes, lindizes):
    t0 = time.time()
    I = float(abs(F.subs({h:miller[0], l:miller[1]}).n())**2)
    print miller, time.time() - t0
    w = pl.sqrt(I/I0/10.)
    amp = pl.sqrt(I/I0)
    #xy = pl.sqrt((ha - miller[0])**2 + (la[:,pl.newaxis] - miller[1])**2)
    Imap += et.gaussian(ha, miller[0], amp, w*float(hmax)/lmax, 0) * et.gaussian(la, miller[1], amp, w, 0)[:,pl.newaxis]


pl.imshow(Imap, extent=(ha[0], ha[-1], la[0], la[-1]), aspect=float(hmax)/lmax)

pl.show()
