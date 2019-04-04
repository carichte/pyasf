#!/usr/bin/env python
"""
Calculate reflection intensities for Ho2PdSi3

Outer Batch file for pyasf.py.

Written by Carsten Richter (carsten.richter@desy.de)

"""
from pyasf import *
from sympy.utilities import lambdify
import itertools
mydict = dict({"Abs":abs})


cs = HoPdSi = unit_cell("HPS_D1.cif", resonant="Ho")
cs.get_tensor_symmetry()
cs.build_unit_cell()


if __name__ == "__main__":
    hlist = []
    klist = []
    llist = []
    thlist = []
    Ilist = []
    
    energy = 24600.
    
    cs.calc_structure_factor(DQ=False, DD=False, subs=True, evaluate=True, Temp=False)
    #theta = pyasf.makefunc(cs.theta.subs(cs.energy, energy))
    theta = pyasf.makefunc(cs.theta.subs(cs.energy, energy)*180/pyasf.sp.pi)
    
    for miller in cs.iter_rec_space(1.5):
        if miller == (0,0,0): continue
        h,k,l = miller
        
        Intens = abs(cs.DAFS(energy, miller, force_refresh=False))**2
        if Intens < 1e-3:
            continue
        thistheta = theta.dictcall(cs.f)
        
        hlist.append(h)
        klist.append(k)
        llist.append(l)
        thlist.append(thistheta)
        Ilist.append(Intens)
        print((miller, thistheta))

    harray = np.array(hlist, dtype=int)
    karray = np.array(klist, dtype=int)
    larray = np.array(llist, dtype=int)
    tharray = np.array(thlist)
    Iarray = np.array(Ilist).squeeze()
    
    ind=np.lexsort(keys = (larray, karray, harray, tharray))
    
    data = np.vstack((harray[ind], karray[ind], larray[ind], tharray[ind], Iarray[ind])).T
    
    np.savetxt("HoPdSi_reflexe_%.1fkeV.dat"%(energy/1000), data, fmt="%i\t%i\t%i\t%2.5f\t%f")


