import os
sum_formula = "LiF"
ciffile = "cif/LiF_18012.cif"

ciffile = os.path.join(os.path.split(__file__)[0], ciffile)

def get_cs():
    import pyasf
    sp = pyasf.sp
    cs = pyasf.unit_cell(ciffile)
    cs.get_tensor_symmetry()
    cs.build_unit_cell()
    return cs