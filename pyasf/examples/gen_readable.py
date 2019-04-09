"""
    script to fetch all space group generators from Bilbao crystallographic server.
"""


import pyasf
import os
import sympy as sp

SGs = range(1,231)
#SGs = range(15,16)

x,y,z = sp.symbols("x y z", real=True)
v = sp.Matrix((x,y,z))

for sg in SGs:
    settings = pyasf.get_ITA_settings(sg)
    for sgsym in sorted(settings):
        trmat = settings[sgsym]
        print((sg, sgsym, trmat))

        #generators = pyasf.fetch_ITA_generators(sg, sgsym)
        generators = pyasf.get_generators(sg, sgsym)
        with open("generators.txt", "a") as fh:
            fh.write(os.linesep)
            fh.write("%i; %s; %s%s"%(sg, sgsym, trmat, os.linesep))
        for gen in generators:
            genv = tuple(gen[:,:3]*v + gen[:,3])
            genv = [str(t).replace(" ", "") for t in genv]
            genv = ", ".join(genv)
            gens = str(gen)
            with open("generators.txt", "a") as fh:
                fh.write("# %25s; %s%s"%(genv, gens, os.linesep))
                #fh.write("%s%s"%(str(genv).strip("()"), os.linesep))


