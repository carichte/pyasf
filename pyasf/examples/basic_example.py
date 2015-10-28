# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pylab
import pyasf
import sympy as sp

# <codecell>

cs = pyasf.unit_cell("LiNbO3_R3cH_Abrahams86_ICSD_61118.cif", resonant="Nb")

# <codecell>

cs.AU_positions # positions in asymmetric unit

# <codecell>

cs.AU_formfactorsDDc["Nb1"] # cartesian representation of dipole dipole form factor tensor in asymmetric unit

# <codecell>

cs.get_tensor_symmetry() # apply all symmetry constraints of space group to all tensors (debye waller U_ij, f_ij)

# <codecell>

cs.AU_formfactorsDDc["Nb1"] # now with symmetry

# <codecell>

cs.U["Nb1"] # symmetry of U is different due to differeny basis

# <codecell>

cs.AU_formfactorsDD["Nb1"] # representation of dipole dipole form factor tensor in crystal basis

# <codecell>

vector = sp.Matrix([1,1,1]) # create random vector
print cs.M * vector # the same vector in cartesian system defined as in Trueblood: doi:10.1107/S0108767396005697

# <codecell>

cs.build_unit_cell() # construct unit cell from asymmetric unit
F0 = cs.calc_structure_factor((0,0,3), Temp=False) # calculate structure factor without temperature

# <codecell>

print F0.n().simplify() # evaluate structure factor ==> forbidden reflection

# <codecell>

cs.transform_structure_factor()
cs.calc_scattered_amplitude() # calculate Structure Factor for higher orders tensors

# <codecell>

print cs.E["ss"] # also zero for sigma sigma scattering

# <codecell>

print cs.E["sp"] # some dipole quadrupole scattering in sigma pi scattering channel

# <codecell>

Fhot = cs.calc_structure_factor((0,0,3), Temp=True) # calculate structure factor with temperature

# <codecell>

print Fhot.n().simplify() # also no temperature scattering

# <codecell>

cs.Q * cs.M * vector # same vector as before but in laboratory system as defined in doi:10.1107/S0108767391011509
# others can be defined...

# <rawcell>

# Now lets check some more basic functionality

# <codecell>

cs.subs # this dictionary contains all (momentary) values

# <codecell>

cs.Uaniso # but here the values for U from the ciffile

# <codecell>

cs.hkl() # current reflection?

# <codecell>

cs.set_temperature # function that can be used to calculate ADPs from debye temperature or einstein temperature once given

# <codecell>

cs.charges

# <codecell>

cs.elements

# <codecell>

cs.occupancy

# <codecell>

cs.positions["Li1"] # all positions in unit cell

# <codecell>

cs.get_density()

# <codecell>

cs.get_stoichiometry()

# <codecell>

cs.get_nearest_neighbors("Li1", 8) # returns labels, distance and difference vector

# <codecell>

cs.get_thermal_ellipsoids("Nb1") # eigenvalues and eigenvectors for ADP of Nb

# <codecell>

cs.multiplicity

# <codecell>

cs.dE # edge shift

# <codecell>

gen = cs.iter_rec_space(0.3) # iterator for all reflections in a ewald volume 2*sin(th)/lambda < 0.3

# <codecell>

print list(gen)

# <codecell>

cs.weights

# <codecell>

cs.metric_tensor

# <codecell>

cs.metric_tensor_inv

# <codecell>

cs.metric_tensor_inv.subs(cs.subs) # the generic way to insert all current values

# <rawcell>

# Units used are: Anstrom and eV

# <codecell>

cs.F_DD

# <codecell>

func = cs.get_reflection_angles((0,0,1), 8000) # get function to calculat angle to surface and azimuth for each reflection

# <codecell>

%pylab inline

# <codecell>

map(degrees, func((0,1,1), 8000, 0)) # (h,k,l), energy, second azimuth

# <rawcell>

# Example to get real intensities:

# <codecell>

cs.DAFS(8048, (0,0,3))

# <codecell>

cs.DAFS(8048, (0,0,6))

# <codecell>

Energy = linspace(17000, 20000, 1001)

# <codecell>

plot(Energy, cs.DAFS(Energy, (0,0,6), Temp=False))
plot(Energy, cs.DAFS(Energy, (0,0,6), Temp=True))

# <codecell>

plot(Energy, cs.DAFS(Energy, (0,2,6), Temp=False))
plot(Energy, cs.DAFS(Energy, (0,2,6), Temp=True))

# <codecell>

for table in ["Henke", "Sasaki", "BrennanCowan", "CromerLiberman"]:
    del cs._ftab
    plot(Energy, cs.DAFS(Energy, (0,0,6), table=table), label=table)
legend()

# <codecell>


# <codecell>


