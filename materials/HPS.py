import os
import numpy as np 

sum_formula = "Ho2PdSi3"
ciffile = "cif/HPS_D1.cif"
ciffile = os.path.join(os.path.split(__file__)[0], ciffile)
#density = 4.63
SpaceGroup = 1
layertypes = ("a", "b", "c", "d")


def add_layer(cs, label, zoffset=0):
    if label.lower() not in layertypes:
        raise ValueError("Layer with label %s not supported"%label)
    fname = os.path.join(os.path.split(__file__)[0], "cif/HPS_%s.txt"%label.upper())
    layer = np.loadtxt(fname, dtype=str)
    for atom in layer:
        label, x, y, z = atom
        z = str(float(z)/8 + zoffset)
        label += str(cs.elements.values().count(label))
        cs.add_atom(label, (x,y,z))


def get_cs(stack = None, AAS=True):
    import pyasf
    sp = pyasf.sp
    if stack==None:
        if AAS:
            cs = pyasf.unit_cell(ciffile, resonant="Ho")
        else:
            cs = pyasf.unit_cell(ciffile)
        fac = 1
    else:
        cs = pyasf.unit_cell(1)
        for i in range(len(stack)):
            add_layer(cs, stack[i], float(i)/len(stack))
        fac = 1./8*len(stack)
        
    
    cs.subs[cs.a] = 8.1
    cs.subs[cs.b] = 8.1
    cs.subs[cs.c] = 32.0 * fac
    cs.subs[cs.alpha] = sp.pi/2
    cs.subs[cs.beta] = sp.pi/2
    cs.subs[cs.gamma] = 2*sp.pi/3
    
    for label in cs.AU_formfactorsDD:
        pyasf.applymethod(cs.AU_formfactorsDD[label], "subs", cs.subs)
        pyasf.applymethod(cs.AU_formfactorsDQ[label], "subs", cs.subs)
        #pyasf.applymethod(cs.AU_formfactorsDDc[label], "subs", cs.subs)
        #pyasf.applymethod(cs.AU_formfactorsDQc[label], "subs", cs.subs)
    
    
    s = cs.subs.copy()
    [s.pop(k) for k in cs.miller]
    
    cs.Gc = cs.Gc.subs(s)
    cs.M = cs.M.subs(cs.subs)
    cs.M0 = cs.M0.subs(cs.subs)
    cs.Minv = cs.Minv.subs(cs.subs)
    cs.M0inv = cs.M0inv.subs(cs.subs)

    
    cs.build_unit_cell()
    return cs