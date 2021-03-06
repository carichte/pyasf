import os
sum_formula = "SrTiO3"
density = 5.13
SpaceGroup = 221
#ciffile = "cif/STO_CollCode80873.cif"
ciffile = "cif/STO_aniso_80873.cif"

ciffile = os.path.join(os.path.split(__file__)[0], ciffile)

def get_cs():
    import pyasf
    sp = pyasf.sp
    cs = pyasf.unit_cell(ciffile, resonant="Sr")
    cs.get_tensor_symmetry()
    cs.build_unit_cell()
    return cs



#def get_cs():
#    import pyasf
#    sp = pyasf.sp
#    np = pyasf.np
#    
#    delta_1 = 0
#    delta_2 = 0
#    delta_3 = 0
#
#    cs = pyasf.unit_cell(SpaceGroup)
#    
#    cs.add_atom("Sr", (0,0,0), 1, dE=-4)
#    cs.add_atom("Ti", (sp.S("1/2"),sp.S("1/2"),sp.S("1/2") - delta_1), 1, dE=14.5)
#    cs.add_atom("O", (sp.S("1/2"),sp.S("1/2"),0 + delta_2), 1)
#    
#    cs.subs[cs.a] = 3.905
#    #cs.subs[cs.c] = 
#    
#    #cs.subs[delta_1] = 0.018
#    #cs.subs[delta_2] = 0.016
#    #cs.subs[delta_3] = 0.015
#    
#    cs.get_tensor_symmetry()
#    cs.build_unit_cell()
#    return cs
