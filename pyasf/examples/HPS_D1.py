#!/usr/bin/env python
"""
Outer Batch file for pyasf.py.

Written by Carsten Richter (carsten.richter@desy.de)

"""
from pyasf import *
from sympy.utilities import lambdify
import itertools
mydict = dict({"Abs":abs})


HoPdSi = unit_cell("HPS_D1.cif", resonant="Ho")

if __name__ == "__main__":
    hlist = []
    klist = []
    llist = []
    thlist = []
    Ilist = []
    
    maxindex = 3
    energy = 24600.
    #indizes = range( -maxindex, maxindex+1)
    indizes = range( 0, maxindex+1)
    
    Intensity, theta = HoPdSi.get_f(energy)
    thetafunc = lambdify(("h", "k", "l"), theta.subs(HoPdSi.subs), ("numpy", mydict))
    Ifunc = lambdify(("h", "k", "l"), Intensity.subs(HoPdSi.subs), ("numpy", mydict))
    for miller in itertools.product(indizes, indizes, indizes):
        if miller == (0,0,0): continue
        h,k,l = miller
        thistheta = np.degrees(thetafunc(h, k, l))
        hlist.append(h)
        klist.append(k)
        llist.append(l)
        thlist.append(thistheta)
        Ilist.append(abs(Ifunc(h,k,l))**2)
        print miller, thistheta
    
    harray = np.array(hlist, dtype=int)
    karray = np.array(klist, dtype=int)
    larray = np.array(llist, dtype=int)
    tharray = np.array(thlist)
    Iarray = np.array(Ilist)
    
    ind=np.lexsort(keys = (larray, karray, harray, tharray))
    
    
    np.savetxt("HoPdSi_reflexe_%.1fkeV.dat"%(energy/1000), np.vstack((harray[ind], karray[ind], larray[ind], tharray[ind], Iarray[ind])).T, fmt="%i\t%i\t%i\t%2.5f\t%f")


