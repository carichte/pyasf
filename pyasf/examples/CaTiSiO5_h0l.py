#!/usr/bin/env python
"""
Cut of Reciprocal Space

Outer Batch file for pyasf.py.

Written by Carsten Richter (carsten.richter@desy.de)

"""
import itertools
import pylab as pl
import pyasf
import time
mydict = dict({"Abs":abs})


cs = pyasf.unit_cell("MyBaseFileName_9837.cif")
cs.get_tensor_symmetry()
cs.build_unit_cell()

energy = 10000.


ha = pl.linspace(-2, 2, 801)*10
la = pl.linspace(-2, 2, 801)*10

Imap = 0 * ha * la[:,pl.newaxis]
cs.calc_structure_factor()
I0 = abs(cs.DAFS(energy, (0,0,0), force_refresh=False))**2

def gaussian(x, x0, amp, w, y0=0):
    return amp*pl.exp(-(x-x0)**2/(2*w**2))+y0

lmax = 0
hmax = 0
for miller in cs.iter_rec_space(2.5, False):
    if miller[1] or not any(miller):
        continue
    hmax = max(hmax, miller[0])
    lmax = max(lmax, miller[2])
    
    F = cs.DAFS(energy, miller, force_refresh=False)
    I = abs(F)**2
    print(miller, I)
    
    w = pl.sqrt(I/I0/10.)
    amp = pl.sqrt(I/I0)
    #xy = pl.sqrt((ha - miller[0])**2 + (la[:,pl.newaxis] - miller[1])**2)
    Imap += gaussian(ha, miller[0], amp, w, 0) * gaussian(la, miller[2], amp, w, 0)[:,pl.newaxis]


pl.imshow(pl.sqrt(Imap), extent=(ha[0], ha[-1], la[0], la[-1]), aspect=float(hmax)/lmax)

pl.show()
