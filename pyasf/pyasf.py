#!/usr/bin/env python
"""
    Module provides symbolical calculation of the anisotropic Structure Factor
    of a Crystal of given Space Group and Asymmetric Unit up to
    dipole-quadrupole (DQ) approximation.
    
    Written by Carsten Richter (carsten.richter@desy.de)
    http://www.desy.de/~crichter
"""

import types
import itertools
import copy
import sympy as sp
import numpy as np
import CifFile
import difflib
from time import time
from functions import *
from fractions import Fraction
from scipy.interpolate import interp1d
from sympy.utilities import lambdify
import elements
#mydict = dict({"Abs":abs})
#epsilon=1.e-10

"""
    TODO:
        consider different space group settings
        add symmetry to DQ tensors
        DQ part imaginary?
"""
dictcall = lambda self, d: self.__call__(*[d[k] for k in self.kw])



def makefunc(expr, mathmodule = "numpy"):
    symbols = filter(lambda x: x.is_Symbol, expr.atoms())
    symbols.sort(key=str)
    func = lambdify(symbols, expr, mathmodule)
    func.kw = symbols
    func.kwstr = map(lambda x: x.name, symbols)
    func.dictcall = types.MethodType(dictcall, func)
    func.__doc__ = str(expr)
    return func


def mkfloat(string):
    assert isinstance(string, str), "Invalid input. Need str."
    string = string.strip()
    if string == ".":
        return 0
    i = string.find("(")
    if i>=0:
        string = string[:i]
    return float(string)

def applymethod(Arr, Method, *args):
    for ele in np.nditer(Arr, flags=["refs_ok"], op_flags=['readwrite']):
        if hasattr(ele.item(), Method):
            ele[...] = getattr(ele.item(), Method)(*args)

def applyfunc(Arr, Func):
    for ele in np.nditer(Arr, flags=["refs_ok"], op_flags=['readwrite']):
        ele[...] = Func(ele.item())

ArraySimp1 = np.vectorize(
    lambda ele:ele.expand(trig=True).rewrite(sp.sin, sp.exp).expand() if hasattr(ele, "expand") else ele,
    doc="Vectorized function to simplify symbolic array elements using rewrite")

ArraySimp2 = np.vectorize(
    lambda ele:ele.rewrite(sp.exp, sp.sin).expand() if hasattr(ele, "rewrite") else ele,
    doc="Vectorized function to simplify symbolic array elements using rewrite")


class unit_cell(object):
    """
        This class represents the unit cell of a crystal in terms of Resonant
        Elastic X-Ray Scattering (REXS).
        After initialization one can:
            - fill it with atoms (self.add_atom) of the asymmetric unit
            - reduce degrees of freedom in their atomic scattering tensors
              by application of the crystals symmetry (self.get_tensor_symmetry)
            - generate all atoms on symmetry-equivalent sites (self.build_unit_cell)
            - calculate the Structure Factor in reciprocal coordinates up to 
              dipolar-quadrupolar contributions (self.F_..., self.calc_structure_factor)
            - transform it into the diffractometer system 
              (self.Fd_..., self.transform_structure_factor)
            - calculate the scattered field amplitus (self.E) at given azimut
              'psi' and glancing angle 'theta' (self.calc_scattered_amplitude)
            - try to simplify the resulting expressions in self.E (self.simplify)
        
        
        Units:
            energy : eV
            lambda = 12.398/energy
            
            unit_cell.q = 2 * sin(theta)/lambda
            
    """
    eps = 10*np.finfo(np.float64).eps
    u = 1.660538921e-27 # atomic mass unit
    DEBUG = False
    def __init__(self, structure, resonant="", **kwargs):
        """
            Initializes the crystals unit cell for a given
            structure in the following steps:
                - retrieve space group generators for the given space group
                - calculate real and reciprocal lattice parameters
                - calculates Matrices B, B_0
                - calculates real and reciprocal metric tensors G, G_r

            Optionally loads a structure from a .cif file.
            See:
                Hall SR, Allen FH, Brown ID (1991).
                "The Crystallographic Information File (CIF):
                 a new standard archive file for crystallography".
                Acta Crystallographica A47 (6): 655-685
            A list or string of 'resonant' scattering atoms can be given.
            
            
            Input parameters:
                structure : either
                                - number of space group
                            or
                                - path to .cif-file
        """
        self.sg_sym = None
        self.symmetries = {}
        self.AU_positions = {} # only asymmetric unit
        self.AU_formfactors = {} # only asymmetric unit, isotropic
        self.AU_formfactorsDD = {} # only asymmetric unit, pure dipolar
        self.AU_formfactorsDQ = {} # only asymmetric unit, dipolar-quadrupolar interference
        self.miller = sp.symbols("h k l", integer=True)
        self.subs = {}
        self.energy = sp.Symbol("epsilon", real=True)
        self.subs.update(dict(zip(self.miller, self.miller)))
        self.S = dict([(s.name, s) for s in self.miller]) # dictionary of all symbols
        self.elements = {}
        self.U = {} # dictionary of anisotropic mean square displacement
        self.dE = {} # dictionary of edge shifts
        self.f0func = {}
        self.f0 = {}
        self.Etab = np.array([])
        self.occupancy = {}
        
        if str(structure).isdigit():
            self._init_lattice(int(structure))
        elif len(structure)==2 and str(structure[0]).isdigit():
            sg_num, self.sg_sym = structure
            self._init_lattice(int(sg_num))
        elif os.path.isfile(structure) and \
             os.path.splitext(structure)[1].lower()==".cif":
            self.load_cif(structure, resonant, **kwargs)
                    
        else:
            raise IOError("Invalid input for structure. Has to be either space group number or path to .cif-file")
            
    
    def _init_lattice(self, sg_num):
        self.sg_num = sg_num
        self.generators=get_generators(self.sg_num, self.sg_sym) # fetch the space group generators
        if self.sg_sym!=None:
            self.transform = transform = get_ITA_settings(self.sg_num)[self.sg_sym]
        metrik = get_cell_parameters(self.sg_num, self.sg_sym)
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.system = metrik
        
        recparam = get_rec_cell_parameters(*metrik[0:6])
        self.ar, self.br, self.cr = recparam[:3] # reciprocal lattice parameters
        self.alphar, self.betar, self.gammar = recparam[3:6] # reciprocal lattice angles
        self.B, self.B_0 = recparam[6:8] # transformation matrices
        self.G = sp.Matrix(self.miller)
        self.Gc = self.B * self.G
        self.Gc.simplify()
        self.q = self.Gc.norm()
        self.qfunc = makefunc(self.q, sp)
        self.metric_tensor, self.metric_tensor_inv = recparam[8:10] # metric tensors
        self.V = sp.sqrt(self.metric_tensor.det())
        
    def add_atom(self, label, position, isotropic=True, assume_complex=True, dE=0, occupancy=1):
        """
            Method to fill the asymmetric unit with atoms.
            
            Inputs:
            -------
                label : string
                    The label of the atom.
                    It has to be unique and to start with the symbold of the
                    chemical element.
                position : iterable of length 3
                    Position of the atom in the basis of the lattice vectors.
                isotropic : bool
                    Defines whether the atom is an isotropic scatterer which
                    is mostly the case far from absorption edges.
                assume_complex : bool
                    Defines whether the scalar atomic scattering amplitude shall
                    be assumed to be complex. This can be left False, since the
                    calculations are symbolic and values, that will be entered
                    later, still can be complex.
                dE : scalar
                    Sets the shift of the absorption edge for this particular
                    atom.
        """
        if type(label) is not str: raise TypeError("Invalid label. Need string.")
        if len(position) is not 3: raise TypeError("Enter 3D position object!")
        position = list(position)
        for i in range(3):
            if isinstance(position[i], str):
                position[i] = sp.S(Fraction(position[i]).limit_denominator(1000))
        #label = label.replace("_", "")
        labeltest = label[0].upper()
        if len(label) > 1:
            labeltest += label[1].lower()
        if labeltest in elements.Z.keys():
            element = labeltest
        elif labeltest[:1] in elements.Z.keys():
            element = labeltest[:1]
        else:
            raise ValueError("Atom label should start with the symbol of the chemical element" + \
                             "Chemical element not found in %s"%label)
        self.elements[label] = element
        self.AU_positions[label] = np.array(position)
        self.dE[label] = dE
        if not self.f0func.has_key(element):
            self.f0func[element] = makefunc(calc_f0(element, self.Gc.norm()), sp)
        self.occupancy[label] = occupancy
        if assume_complex:
            my_ff_args = dict({"complex":True})
        else:
            my_ff_args = dict({"real":True})
        if isotropic:
            Sym = sp.Symbol("f_" + label, **my_ff_args)
            self.S[Sym.name] = Sym
            self.AU_formfactors[label] = Sym
        elif not isotropic:
            Sym = sp.Symbol("f_" + label + "_0", real=True)
            self.S[Sym.name] = Sym
            self.AU_formfactors[label] = Sym
            self.AU_formfactorsDD[label] = np.zeros((3,3), dtype=object)
            ind = range(3)
            for i,j in itertools.product(ind, ind):
                if i<=j: 
                    Sym = sp.Symbol("f_%s_dd_%i%i"%(label, i+1, j+1), **my_ff_args)
                    self.S[Sym.name] = Sym
                    self.AU_formfactorsDD[label][i,j] = Sym
                    if i<j:
                        self.AU_formfactorsDD[label][j,i] = Sym
            self.AU_formfactorsDQ[label] = np.zeros((3,3,3), dtype=object)
            kindices=("x", "y", "z")
            for h,i,j in itertools.product(ind, ind, ind):
                if j<=h:
                    Sym = sp.Symbol("f_%s_dq_%s%i%i"%(label, kindices[h], i+1, j+1), **my_ff_args)
                    self.S[Sym.name] = Sym
                    self.AU_formfactorsDQ[label][h,i,j] = Sym
                    if j<h:
                        self.AU_formfactorsDQ[label][j,i,h] = Sym 
                        # See eq. 12 in Kokubun et al. http://dx.doi.org/10.1103/PhysRevB.82.205206
            if self.DEBUG:
                print(self.AU_formfactorsDD[label], self.AU_formfactorsDQ[label])
        else: raise TypeError("Enter boolean argument for isotropic")
    
    def load_cif(self, fname, resonant="", max_denominator=10000):
        """
            Loads a structure from a .cif file.
            See:
                Hall SR, Allen FH, Brown ID (1991).
                "The Crystallographic Information File (CIF): a new standard archive file for crystallography".
                Acta Crystallographica A47 (6): 655-685
            
            A list or string of resonant scattering atoms can be given.
            
        """
        try:
            cf = CifFile.ReadCif(fname)
            self.cif = cif = cf.first_block()
        except Exception as e:
            print("File doesn't seem to be a valid .cif file: %s"%fname)
            raise IOError(e)
        try:
            sg_num = int(cif["_symmetry_int_tables_number"])
        except Exception as errmsg:
            raise ValueError("space group number could not be determined from .cif file `%s`:\n%s"%(structure, errmsg))
        ITA = get_ITA_settings(sg_num)
        if len(ITA)>1:
            print("Multiple settings found in space group %i"%sg_num)
            if cif.has_key("_symmetry_space_group_name_h-m"):
                sg_sym = cif["_symmetry_space_group_name_h-m"]
                sg_sym = "".join(sg_sym.split())
                if sg_sym.endswith("S"):
                    sg_sym = sg_sym[:-1] + ":1"
                if sg_sym.endswith("Z"):
                    sg_sym = sg_sym[:-1] + ":2"
                    
                settings = ITA.keys()
                ratios = [difflib.SequenceMatcher(a=sg_sym, b=set).ratio() for set in settings]
                setting = settings[np.argmax(ratios)]
                print("  Identified symbol `%s' from .cif entry `%s'"\
                      %(setting, sg_sym))
                self.sg_sym = sg_sym = setting
            
        
        def getcoord(cifline):
            coord = mkfloat(cifline)
            coord = Fraction(coord)
            coord = coord.limit_denominator(max_denominator)
            return coord
        def getangle(cifline):
            ang = mkfloat(cif[cifline])/180
            ang = Fraction("%.15f"%ang)
            ang = sp.S(ang.limit_denominator(max_denominator)) * sp.pi
            return ang
        
        self._init_lattice(sg_num)
        
        self.cifinfo = {}
        if cif.has_key("_chemical_formula_sum"):
            self.cifinfo["SumFormula"] = cif["_chemical_formula_sum"]
        self.subs[self.a] = mkfloat(self.cif["_cell_length_a"])
        self.subs[self.b] = mkfloat(self.cif["_cell_length_b"])
        self.subs[self.c] = mkfloat(self.cif["_cell_length_c"])
        if self.alpha.is_Symbol:
            self.subs[self.alpha] = getangle("_cell_angle_alpha")
        if self.beta.is_Symbol:
            self.subs[self.beta]  = getangle("_cell_angle_beta")
        if self.gamma.is_Symbol:
            self.subs[self.gamma] = getangle("_cell_angle_gamma")
        
        if not cif.has_key("_atom_site_label"):
            raise ValueError("No atoms found in .cif file")
        loop = cif.GetLoop("_atom_site_label")
        for line in loop:
            symbol = line._atom_site_type_symbol
            symbol = filter(str.isalpha, symbol)
            label = line._atom_site_label
            px = sp.S(getcoord(line._atom_site_fract_x))
            py = sp.S(getcoord(line._atom_site_fract_y))
            pz = sp.S(getcoord(line._atom_site_fract_z))
            occ = mkfloat(line._atom_site_occupancy)
            position = (px, py, pz)
            isotropic = (symbol not in resonant)
            if loop.has_key("_atom_site_U_iso_or_equiv"):
                iso = mkfloat(line._atom_site_U_iso_or_equiv)
            elif loop.has_key("_atom_site_B_iso_or_equiv"):
                iso = mkfloat(line._atom_site_B_iso_or_equiv)
                iso /= 8. * np.pi**2
            else:
                iso = 0
            
            self.U[label] = sp.eye(3) * iso
            self.add_atom(label, position, isotropic, assume_complex=True, occupancy=occ)
        
        if cif.has_key("_atom_site_aniso_label"):
            loop = cif.GetLoop("_atom_site_aniso_label")
            for num, line in enumerate(loop):
                label = line._atom_site_aniso_label
                for (i,j) in [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]:
                    kw_U = "_atom_site_aniso_U_%i%i"%(i,j)
                    kw_B = "_atom_site_aniso_B_%i%i"%(i,j)
                    if loop.has_key(kw_U):
                        value = mkfloat(loop[kw_U][num])
                    elif loop.has_key(kw_U):
                        value = mkfloat(loop[kw_B][num])
                        value /= 8. * np.pi**2
                    self.U[label][i-1,j-1] = self.U[label][j-1,i-1] = value

    
    
    def find_symmetry(self, start_sg=230):
        """
            Not yet perfect
        """
        element_pos = {}
        for label in self.AU_positions.keys():
            element = self.elements[label]
            if element_pos.has_key(element):
                element_pos[element].append(self.AU_positions[label])
            else:
                element_pos[element] = [self.AU_positions[label]]
        if self.DEBUG:
            print element_pos
        while start_sg>0:
            print("Trying space group %i..."%start_sg)
            UCtest = unit_cell(start_sg)
            skipSG = False
            for element in element_pos.keys():
                positions = np.array(copy.deepcopy(element_pos[element]))
                num = 1
                while len(positions)>0 and not skipSG:
                    UCtest.add_atom(element+str(num), positions[0])
                    UCtest.build_unit_cell()
                    for newpos in UCtest.positions[element+str(num)]:
                        ind = (((newpos - np.array(positions))**2).sum(1))<UCtest.mindist
                        if sum(ind)==1:
                            if self.DEBUG:
                                print "Found", newpos, "in", positions
                            positions = positions[~ind]
                        else:
                            print "Not found", newpos
                            skipSG = True
                            break
                    num+=1
                if skipSG: break
            if skipSG:
                start_sg-=1
            else:
                break
        return UCtest
    
    def get_tensor_symmetry(self, labels = None):
        """
            Applies Site Symmetries of the Space Group to the Scattering Tensors.
        """
        if labels == None:
            labels = self.AU_formfactorsDD.keys()
        self.equations={}
        for label in labels:
            equations=[]
            for generator in self.generators:
                W = np.array(generator[:,:3])
                w = np.array(generator[:,3]).ravel()
                if self.DEBUG: print w, W
                new_position = full_transform(W, self.AU_positions[label]) + w
                if self.DEBUG: print new_position
                new_position = np.array([stay_in_UC(i) for i in new_position])
                #if (new_position == self.AU_positions[label]).all(): #1.8.13
                if (new_position - self.AU_positions[label]).sum() < self.eps:
                    new_formfactorDD = full_transform(W, self.AU_formfactorsDD[label])
                    new_formfactorDQ = full_transform(W, self.AU_formfactorsDQ[label])
                    for eq in (self.AU_formfactorsDD[label] - new_formfactorDD).ravel():
                        if eq not in equations and hasattr(eq, "is_number") and not eq.is_number:
                            equations.append(eq)
                    for eq in (self.AU_formfactorsDQ[label] - new_formfactorDQ).ravel():
                        if eq not in equations and hasattr(eq, "is_number") and not eq.is_number:
                            equations.append(eq)
            if self.DEBUG:
                print label, equations
            self.equations[label] = equations
            symmetries=sp.solve(equations, dict=True)
            if not symmetries:
                continue
            assert len(symmetries)==1, "Unusual length of result of sp.solve (>1)"
            symmetries = symmetries[0]
            self.symmetries[label]=symmetries
            if self.DEBUG:
                print symmetries
            applymethod(self.AU_formfactorsDD[label], "subs", symmetries)
            applymethod(self.AU_formfactorsDQ[label], "subs", symmetries)
            if self.DEBUG:
                print self.AU_formfactorsDD[label], self.AU_formfactorsDQ[label]
    
    
    def _transform(self, generator, AU=False):
        """
            transform structure unit with a given generator.
        """
        if AU:
            positions = self.AU_positions
            formfactorsDD = self.AU_formfactorsDD
            formfactorsDQ = self.AU_formfactorsDQ
        else:
            positions = self.positions
            formfactorsDD = self.formfactorsDD
            formfactorsDQ = self.formfactorsDQ
        if np.shape(generator)==(4, 3):
            generator = np.array(generator).T
        if np.shape(generator)==(3, 4):
            W = np.array(generator[:,:3])
            w = np.ravel(generator[:,3])
        elif np.shape(generator)==(3, 3):
            W = np.array(generator[:,:3])
            w = np.zeros(3)
        elif len(np.ravel(generator))==3:
            W = np.diag((1,1,1))
            w = np.ravel(generator)
        else:
            return
        for name in positions.keys():
            if AU:
                new_position = full_transform(W, positions[name]) + w
                new_position = new_position%1
                positions[name] = new_position
                if formfactorsDD.has_key(name):
                    formfactorsDD[name] = full_transform(W, formfactorsDD[name])
                if formfactorsDQ.has_key(name):
                    formfactorsDQ[name] = full_transform(W, formfactorsDQ[name])
            else:
                for i in range(len(positions[name])):
                    new_position = full_transform(W, positions[name][i]) + w
                    new_position = new_position%1
                    positions[name][i] = new_position
                    if formfactorsDD.has_key(name):
                        formfactorsDD[name][i] = full_transform(W, formfactorsDD[name][i])
                    if formfactorsDQ.has_key(name):
                        formfactorsDQ[name][i] = full_transform(W, formfactorsDQ[name][i])
    
    def build_unit_cell(self):
        """
            Generates all Atoms of the Unit Cell.
        """
        #if hasattr(self, "cell"): return
        self.cell=[]
        self.positions={}
        self.formfactors={}
        self.formfactorsDD={}
        self.formfactorsDQ={}
        for name in self.AU_positions.keys():
            self.positions[name]=[]
            self.formfactors[name]=[]
            if self.AU_formfactorsDD.has_key(name):
                self.formfactorsDD[name]=[]
            if self.AU_formfactorsDQ.has_key(name):
                self.formfactorsDQ[name]=[]
            for generator in self.generators:
                W = np.array(generator[:,:3])
                w = np.array(generator[:,3]).ravel()
                #print W, self.AU_positions[name], w
                new_position = full_transform(W, self.AU_positions[name]) + w
                new_position = np.array([stay_in_UC(i) for i in new_position])
                if len(self.positions[name])>0: 
                    ind = ((new_position - np.array(self.positions[name]))**2).sum(1)
                else:
                    ind = np.array((1))
                if not (ind<self.eps).any():
                    #if new_position not in self.positions[name]:
                    if self.AU_formfactors.has_key(name): 
                        self.formfactors[name].append(self.AU_formfactors[name])
                        self.cell.append((new_position, self.formfactors[name][-1]))
                    if self.AU_formfactorsDD.has_key(name):
                        self.formfactorsDD[name].append(full_transform(W, self.AU_formfactorsDD[name]))
                        self.cell.append((new_position, self.formfactorsDD[name][-1]))
                    if self.AU_formfactorsDQ.has_key(name):
                        self.formfactorsDQ[name].append(full_transform(W, self.AU_formfactorsDQ[name]))
                        self.cell.append((new_position, self.formfactorsDQ[name][-1]))
                    self.positions[name].append(new_position)
        # rough estimate of the infimum of the distance of atoms:
        self.mindist = 1./len(self.cell)**(1./3)/1000.
    
    
    def get_density(self):
        """
            Calculates the density in g/cm^3 from the structure.
        """
        import pyxrr.xray_interactions as xi
        assert hasattr(self, "positions"), \
            "Unable to find atom positions. Did you forget to perform the unit_cell.build_unit_cell() method?"
        self.species = np.unique(self.elements.values())
        self.amounts = dict(zip(self.species, np.zeros(len(self.species), dtype=int)))
        for label in self.positions.iterkeys():
            self.amounts[self.elements[label]] += len(self.positions[label])
        
        
        self.weights = dict([(atom, xi.get_element(atom)[1]) for atom in self.species])
        div = gcd(*self.amounts.values())
        components = ["%s%i"%(item[0], item[1]/div) for item in self.amounts.iteritems()]
        components = map(lambda x: x[:-1] if (x[-1]=="1" and not x[-2].isdigit()) else x, components)
        self.SumFormula = "".join(components)
        
        
        self.total_weight = sum([self.weights[atom]*self.amounts[atom] for atom in self.species])
        self.density = self.u * self.total_weight/1000. / (self.V*1e-10**3) # density in g/cm^3
        return float(self.density.subs(self.subs).n())
    
    def calc_structure_factor(self, miller=None, DD=True, DQ=True, Temp=True):
        """
            Takes the three Miller-indices of Type int and calculates the Structure Factor in the reciprocal basis.
        """
        if miller==None:
            miller = self.miller
        self.subs.update(zip(self.miller, miller))
        if not hasattr(self, "cell"):
            self.build_unit_cell()
        G = self.G.subs(self.subs)
        Gc = self.Gc.subs(self.subs)
        self.F_0 = sp.S(0)
        self.F_DD = np.zeros((3,3), object)
        self.F_DQ = np.zeros((3,3,3), object)
        for Atom in self.positions.keys(): # get position, formfactor and symmetry if DQ tensor
            if Temp:
                Temp = float(Temp)
                DW = sp.exp(-2*sp.pi**2 * Temp * Gc.dot(self.U[Atom].dot(Gc)))
            else:
                DW = 1
            if self.formfactors.has_key(Atom):
                for i in range(len(self.formfactors[Atom])):
                    r = self.positions[Atom][i]
                    f = self.formfactors[Atom][i]
                    o = self.occupancy[Atom]
                    if self.DEBUG: print r, G.dot(r)
                    self.F_0 += o * f * sp.exp(2*sp.pi*sp.I * G.dot(r)) * DW
            if self.formfactorsDD.has_key(Atom) and DD:
                for i in range(len(self.formfactorsDD[Atom])):
                    r = self.positions[Atom][i]
                    f = self.formfactorsDD[Atom][i]
                    self.F_DD += o * f * sp.exp(2*sp.pi*sp.I * G.dot(r)) * DW
            if self.formfactorsDQ.has_key(Atom) and DQ:
                for i in range(len(self.formfactorsDQ[Atom])):
                    r = self.positions[Atom][i]
                    f = self.formfactorsDQ[Atom][i]
                    self.F_DQ += o * f * sp.exp(2*sp.pi*sp.I * G.dot(r)) * DW
        
        self.q = self.qfunc.dictcall(self.subs)
        self.d = 1/self.q
        self.theta = sp.asin(12398./(2*self.d*self.energy))
        
        # simplify:
        if DD:
            self.F_DD = ArraySimp2(self.F_DD)
        if DQ:
            self.F_DQ = ArraySimp2(self.F_DQ)
        
        
    def add_equivalent_SF(self, symop):
        """
            Applies Site Symmetries of the Space Group to the Structure Tensor.
        """
        W = np.array(symop)
        #print (W, self.formfactorsDD[name])
        new_F_DD = full_transform(W, self.F_DD).copy()
        new_F_DQ = full_transform(W, self.F_DQ).copy()
        self.F_DD += new_F_DD
        self.F_DQ += new_F_DQ

    
    def hkl(self):
        return tuple(self.subs[ind] for ind in self.miller)
    
    def transform_structure_factor(self, AAS = True, simplify=True, subs=True):
        """
            First transforms F to a real space, cartesian, crystal-fixed system ->Fc.
            Then transforms Fc to the diffractometer system, which is G along xd and sigma along zd ->Fd.
            
            This happence according to the work reported in:
            Acta Cryst. (1991). A47, 180-195 [doi:10.1107/S010876739001159X]
        """
        if not hasattr(self, "F_0") or \
           (AAS and (not hasattr(self, "F_DD") or not hasattr(self, "F_DQ"))):
            raise ValueError("No Reflection initialized. "
                             " Run self.calc_structure_factor() first.")
        if AAS:
            B_inv_T = np.array(self.B.T.inv())
            B_0_inv_T = np.array(self.B_0.T.inv())
        # now the structure factors in a cartesian system follow
        self.Fc_0 = self.F_0
        # self.Fc_DD = full_transform(B_inv_T, self.F_DD) #rethink that :-/
        # self.Fc_DQ = full_transform(B_inv_T, self.F_DQ)
        if AAS:
            self.Fc_DD = full_transform(B_0_inv_T, self.F_DD)
            self.Fc_DQ = full_transform(B_0_inv_T, self.F_DQ)
        
        # and now: the rotation into the diffractometer system 
        #       (means rotation G into xd-direction)
        # calculate corresponding angles
        
        Gc = self.Gc.subs(zip(self.miller, self.hkl()))
        
        if Gc[1] == 0: phi = 0
        elif Gc[0] == 0: phi = sp.S("pi/2")*sp.sign(Gc[1])
        else: phi = sp.atan(Gc[1]/Gc[0])
        if Gc[2] == 0: xi = 0
        elif Gc[0] == 0 and Gc[1] == 0: xi = sp.S("pi/2")*sp.sign(Gc[2])
        else: xi = sp.atan(Gc[2]/sp.sqrt(Gc[0]**2 + Gc[1]**2))
        
        if simplify:
            if hasattr(xi,  "simplify"):
                xi  = xi.simplify()
            if hasattr(phi, "simplify"):
                phi = phi.simplify()
        self.xi, self.phi = xi, phi
        
        # introduce rotational matrices
        #  Drehung um z in -phi Richtung
        self.Phi = np.array([[ sp.cos(phi), sp.sin(phi), 0], 
                             [-sp.sin(phi), sp.cos(phi), 0], 
                             [0,                      0, 1]]) 
        #  Drehung um y in xi Richtung
        self.Xi  = np.array([[ sp.cos(xi), 0, sp.sin(xi)],
                             [          0, 1,          0], 
                             [-sp.sin(xi), 0, sp.cos(xi)]])
        
        # combined rotation
        self.Q = np.dot(self.Xi, self.Phi)
        if subs:
            applymethod(self.Q, "subs", self.subs)
            applymethod(self.Q, "n")
        if simplify:
            applymethod(self.Q, "simplify")
            if AAS:
                self.Fc_DD = ArraySimp1(self.Fc_DD)
                self.Fc_DQ = ArraySimp1(self.Fc_DQ)

        self.Fd_0 = self.Fc_0
        if AAS:
            self.Fd_DD = full_transform(self.Q, self.Fc_DD)
            self.Fd_DQ = full_transform(self.Q, self.Fc_DQ)
        self.Q = sp.Matrix(self.Q)
        self.Gd = self.Q * Gc
        
    def transform_rec_lat_vec(self, miller, psi=0, inv=False):
        assert len(miller)==3, "Input has to be vector of length 3."
        miller = sp.Matrix(miller)
        if not hasattr(self, "Q"):
            self.transform_structure_factor()
        UB = self.Q * self.B
        if psi!=0:
            Psi = np.array([[1, 0, 0], 
                           [0,sp.cos(psi),sp.sin(psi)], 
                           [0, -sp.sin(psi), sp.cos(psi)]])
            UB = sp.Matrix(Psi) * UB
        if inv:
            UB = UB.inv()
        return UB * miller
    
    def theta_degrees(self, energy=None, h=None, k=None, l=None):
        """
            Returns the Bragg angle (theta) in degree for a given energy in eV.
        """
        if energy!=None:
            self.subs[self.energy] = energy
        subs = zip(self.G, (h,k,l))
        subs = filter(lambda x: x[1]!=None, subs)
        subs = dict(subs)
        return sp.N(self.theta.subs(self.subs).subs(subs) * 180/sp.pi)
    
    def calc_scattered_amplitude(self, psi=None, assume_imag=True, 
                                       DD=True, DQ=True, simplify=True):
        if not hasattr(self, "Fd_0"):
            self.transform_structure_factor()
        if psi==None:
            psi = sp.Symbol("psi", real=True)
            self.S[psi.name] = psi
        self.transform_structure_factor()
        sigma = sp.Matrix([0,0,1])
        k = self.energy / 12398.42
        self.k_plus = 2 * k * sp.cos(self.theta)
        pi_i = sp.Matrix([sp.cos(self.theta), sp.sin(self.theta), 0])
        pi_s = sp.Matrix([sp.cos(self.theta), -sp.sin(self.theta), 0])
        #vec_k_i = k * np.array([-sp.sin(theta), sp.cos(theta), 0]) #alt
        #vec_k_s = k * np.array([sp.sin(theta), sp.cos(theta), 0]) #alt
        
        # introduce rotational matrix
        Psi = np.array([[1, 0, 0], 
                       [0,sp.cos(psi),sp.sin(psi)], 
                       [0, -sp.sin(psi), sp.cos(psi)]])
        self.Fd_psi_0 = self.Fd_0.simplify()
        if DD:
            self.Fd_psi_DD = full_transform(Psi, self.Fd_DD)
            if simplify:
                applymethod(self.Fd_psi_DD, "simplify")
        if DQ:
            self.Fd_psi_DQ = full_transform(Psi, self.Fd_DQ)
            if simplify:
                applymethod(self.Fd_psi_DQ, "simplify")
        self.Gd_psi = sp.Matrix(Psi) * self.Gd
        if self.DEBUG:
            print self.Gd, self.Gd_psi
        
        # calculate symmetric and antisymmetric DQ Tensors
        if not assume_imag and DQ:
            self.Fd_psi_DQs = (self.Fd_psi_DQ - self.Fd_psi_DQ.transpose(0,2,1).conjugate())/2.
            self.Fd_psi_DQa = (self.Fd_psi_DQ + self.Fd_psi_DQ.transpose(0,2,1).conjugate())/2.
        elif DQ:
            self.Fd_psi_DQs = (self.Fd_psi_DQ + self.Fd_psi_DQ.transpose(0,2,1))/2.
            self.Fd_psi_DQa = (self.Fd_psi_DQ - self.Fd_psi_DQ.transpose(0,2,1))/2.
        

        self.Fd = np.eye(3) * self.Fd_psi_0
        if DD:
            self.Fd += self.Fd_psi_DD
        if DQ:
            self.Fd += sp.I * self.q      * self.Fd_psi_DQs[0] \
                     + sp.I * self.k_plus * self.Fd_psi_DQa[1]
        
        #self.Fd = (np.eye(3) * self.Fd_psi_0 + self.Fd_psi_DD + sp.I*self.q*self.Fd_psi_DQs[0] + sp.I*self.k_plus*self.Fd_psi_DQa[1])
        
        #self.Fd_alt = ( np.eye(3) * self.Fd_psi_0 + self.Fd_psi_DD + sp.I*np.tensordot(self.Fd_psi_DQ, vec_k_i, axes=(0,0))
        #                         + sp.I*np.tensordot(self.Fd_psi_DQ.transpose(0,2,1).conjugate(), vec_k_s, axes=(0,0)) )
        
        self.Fd = sp.Matrix(self.Fd)
        if simplify:
            self.Fd.simplify()
        #self.Fd_alt = sp.Matrix(sp.expand(self.Fd_alt))
        
        
        self.E = {}
        self.E["ss"] = (sigma.T * self.Fd * sigma)[0]
        self.E["sp"] = (pi_s.T * self.Fd * sigma)[0]
        self.E["ps"] = (sigma.T * self.Fd * pi_i)[0]
        self.E["pp"] = (pi_s.T * self.Fd * pi_i)[0]
        
        #return self.E
    
    def DAFS(self, energy, miller=None, f1=None, f2=None, DD=False, DQ=False,
             fwhm_ev=0, func_output=False, equivalent=False, table="Sasaki",
             Temp=True):
        """
            Calculates a Diffraction Anomalous Fine Structure (DAFS) curve for
            a given array of energies and a certain Bragg reflection hkl
            specified by the miller indices 3-tuple.
            The spectral fine structure of atoms is retrieved from a database
            specified in ``table``.
            Per default, dipole-dipole (DD) term as well as dipole-quadrupole
            (DQ) term are neglected.
            Alternatively, the atomic fine structure f1/f2 can be given for
            each site specified via {label:array} dictionary.
            They ``fwhm_ev`` keyword sets the resolution of the dispersion
            correction (f1, f2) when calculated via deltaf package by using
            ``table='deltaf'``. See the deltaf docstring for more information.
            If atoms of the same species shall be considered to be equivalent,
            the same fine structures are used.
            
            If ``func_output`` is True, a function for the structure amplitude
            F(E) is returned. Otherwise, it's the Intensity array.
        """
        import pyxrr.xray_interactions as xi
        self.f = dict({})
        if f1==None:
            f1=dict({})
        else:
            for key in f1.iterkeys():
                assert key in self.AU_formfactors.keys(), \
                    "Label %s not found in present unit cell"%key
            f1 = f1.copy()
        if f2==None:
            f2=dict({})
        else:
            for key in f2.iterkeys():
                assert key in self.AU_formfactors.keys(), \
                    "Label %s not found in present unit cell"%key
            f2 = f2.copy()
        if (miller!=None and len(miller)==3) or not hasattr(self, "F_0"):
            self.calc_structure_factor(miller, DD=DD, DQ=DQ, Temp=Temp)
            self.transform_structure_factor(AAS=(DQ+DD))
            self.f0.clear()
        if not np.all(energy == self.Etab):
            self.f1tab = {}
            self.f2tab = {}
        for label in self.AU_formfactors:
            ffsymbol = self.AU_formfactors[label]
            element = self.elements[label]
            
            Z = elements.Z[element]
            
            if not self.f0.has_key(element):
                if self.DEBUG: print("Calculating nonresonant scattering amplitude for %s"%element)
                self.f0[element] = self.f0func[element].dictcall(self.subs)
            
            if equivalent:
                tmp = sp.Symbol("f_" + element)
                self.subs[ffsymbol] = tmp - Z + self.f0[element]
                ffsymbol = tmp
                if self.f1tab.has_key(element) and self.f2tab.has_key(element):
                    continue
            else:
                self.subs[ffsymbol] = ffsymbol - Z + self.f0[element]
            
            if f1.has_key(label):
                f1[ffsymbol] = f1[label]
                del f1[label]
            if f2.has_key(label):
                f2[ffsymbol] = f2[label]
                del f2[label]
            if not f1.has_key(ffsymbol) or not f2.has_key(ffsymbol):
                # then take dispersion correction for isolated atom
                dE = self.dE[label] # edge shift in eV
                if self.f1tab.has_key(element) and self.f2tab.has_key(element):
                    f1tab = self.f1tab[element]
                    f2tab = self.f2tab[element]
                else:
                    #print("Fetching tabulated dispersion correction values for %s from %s"%(element, table))
                    if table=="deltaf":
                        import deltaf
                        # here the dispersion correction is calculated
                        #f1tab, f2tab = deltaf.getfquad(element, energy - self.dE[name], fwhm_ev)
                        f1tab = deltaf.getfquad(element, energy - dE, fwhm_ev, f1f2="f1")
                        f2tab = deltaf.getfquad(element, energy - dE, fwhm_ev, f1f2="f2")
                        f1tab += Z
                    else:
                        # take dispersion correction from database
                        f1tab, f2tab = xi.get_f1f2_from_db(element, energy - dE, table=table)
                    self.f1tab[element] = f1tab
                    self.f2tab[element] = f2tab
                    self.Etab = energy.copy()
                    #print("Done.")
                if not f1.has_key(ffsymbol):
                    f1[ffsymbol] = f1tab
                if not f2.has_key(ffsymbol):
                    f2[ffsymbol] = f2tab
            #if self.subs.has_key(ffsymbol): self.subs.pop(ffsymbol)
            #print ffsymbol, f1[ffsymbol].shape, f2[ffsymbol].shape
            self.f[ffsymbol] = np.array(f1[ffsymbol] + 1j*f2[ffsymbol], dtype=complex)
        
        Feval = self.F_0.subs(self.subs).n()
        
        F0_func = makefunc(Feval)
        
        if func_output:
            return F0_func
        else:
            return abs(F0_func.dictcall(self.f))**2
    
    def simplify(self): # a longshot
        if not hasattr(self, "E"): return
        for pol in self.E:
            currlen = sp.count_ops(str(self.E[pol]))
            for i in range(3):
                print i
                for new in (sp.expand_mul(self.E[pol]), self.E[pol].expand(trig=True), sp.expand_complex(self.E[pol]).expand()):#, sp.trigsimp(self.E[pol])):
                    print currlen
                    if sp.count_ops(str(new))<currlen:
                        currlen=sp.count_ops(str(new))
                        self.E[pol] = new
        return self.E
    
    def eval_AAS(self, energy=None, table="Sasaki"):
        import pyxrr.xray_interactions as xi
        if energy!=None:
            self.subs[self.energy] = energy
        else:
            energy = float(self.subs[self.energy])
        self.d = 1./self.qfunc.dictcall(self.subs)
        for label in self.AU_formfactors.iterkeys():
            ffsymbol = self.AU_formfactors[label]
            element = self.elements[label]
            Z = elements.Z[element]
            f0 = self.f0func[element].dictcall(self.subs)
            if ffsymbol.name.endswith("_0"):
                self.subs[ffsymbol] = f0
            else:
                dE = self.dE[label] # edge shift in eV
                try:
                    f_res = xi.get_f1f2_from_db(element, energy - dE, table=table)
                except:
                    f_res = xi.get_f1f2_from_db(element, energy - dE, table = "Henke")
                self.subs[ffsymbol] = sp.S(complex(f_res[0], f_res[1])) - Z + f0

        self.I = dict()
        self.AAS = dict()
        for pol in self.E.iterkeys():
            channel = self.E[pol].expand()
            channel = channel.subs(self.subs)
            #channel = channel.trigsimp()
            channel = channel.expand(Trig=True)
            channel = channel.n()
            self.AAS[pol] = channel
            Intensity = abs(channel)**2
            Intensity = Intensity.expand()
            self.I[pol] = makefunc(Intensity, "numpy")
        

    def get_F0(self, miller=None, energy=None, resonant=True, table="Sasaki", 
                     equivalent=False, Temp=False):
        import pyxrr.xray_interactions as xi
        if energy!=None:
            self.subs[self.energy] = energy
        else:
            energy = float(self.subs[self.energy])
        self.calc_structure_factor(miller, Temp=Temp)
        self.transform_structure_factor(AAS=False)
        
        done = []
        for label in self.AU_formfactors.iterkeys():
            ffsymbol = self.AU_formfactors[label]
            element = self.elements[label]
            if equivalent:
                tmp = sp.Symbol("f_" + element)
                self.subs[ffsymbol] = tmp
                ffsymbol = tmp
                self.S[ffsymbol.name] = ffsymbol
            if ffsymbol in done:
                continue
            
            
            if self.DEBUG:
                print("Calculating nonresonant f_0 for %s..." %ffsymbol.name)
            Z = elements.Z[element]
            f0 = self.f0func[element].dictcall(self.subs)

            if ffsymbol.name.endswith("_0"):
                self.subs[ffsymbol] = f0
            else:
                dE = self.dE[label] # edge shift in eV
                try:
                    f_res = xi.get_f1f2_from_db(element, energy - dE, table=table)
                except:
                    f_res = xi.get_f1f2_from_db(element, energy - dE, table = "Henke")
                if resonant:
                    self.subs[ffsymbol] = sp.S(complex(f_res[0], f_res[1])) - Z + f0
                else:
                    self.subs[ffsymbol] = f0
            done.append(ffsymbol)
        return self.F_0.subs(self.subs).subs(self.subs)
        

    def calc_reflection_angles(self, orientation, energy = None, **kwargs):
        """
            Calculates the angles of Bragg reflections relative to the surface
            of a crystal for given orientation (cut).
            Returns a (theta, psi)-tuple, where theta is the angle between the
            incident beam and the surface and psi is the azimuthal position.
        """
        if energy==None:
            if self.subs.has_key(self.energy):
                self.subs.pop(self.energy)
        else:
            self.subs[self.energy] = energy
        miller = list(sp.symbols("h,k,l", integer=True))
        for i in range(3):
            if kwargs.has_key(miller[i].name):
                miller[i] = kwargs[miller[i].name]
        
        psi2 = sp.Symbol("psi_2", real=True)
        self.S[psi2.name] = psi2
        
        self.calc_structure_factor(orientation, Temp=False)
        self.transform_structure_factor(AAS=False)
        Q = self.Q
        Gc = self.Gc.subs(self.subs).normalized()
        #sintheta = sp.sin(self.theta)
        #costheta = sp.sqrt(1 - sintheta**2)
        #vec_ki_d = sp.Matrix([-sintheta, costheta, 0])
        #self.vec_ki_d = vec_ki_d
        #ref_d = vec_ki_d.cross(self.Gd).T  # Reference vector for psi=0
        ref_d = sp.Matrix([0,0,1]) # Reference vector for psi=0
        ref_c = Q.T * ref_d
        ref_c.simplify()
        self.calc_structure_factor(miller, Temp=False) # secondary reflection
        self.transform_structure_factor(AAS=False)
        Q2 = self.Q
        Q2.simplify()
        sintheta2 = sp.sin(self.theta)
        costheta2 = sp.sqrt(1 - sintheta2**2)
        vec_ki2_d = sp.Matrix([-sintheta2, costheta2, 0])
        ref_d2 = Q2 * ref_c
        Gd = Q2 * Gc
        Psi2 = sp.Matrix([[1,             0,           0],
                          [0,  sp.cos(psi2), sp.sin(psi2)],
                          [0, -sp.sin(psi2), sp.cos(psi2)]])
        ref_d2_psi = Psi2 * ref_d2
        Gd_psi = Psi2 * Gd
        self.Gd_psi = Gd_psi
        newref = vec_ki2_d.cross(Gd_psi) # new mark
        #psi is the angle between ref_d2_psi and newref
        cospsi = newref.dot(ref_d2_psi)/(newref.norm()*ref_d2_psi.norm())
        sintheta = vec_ki2_d.dot(Gd_psi)/(vec_ki2_d.norm()*Gd_psi.norm())
        return sp.asin(sintheta), sp.acos(cospsi)
        
    
    def get_reflection_angles(self, orientation, energy=None):
        """
            For a given surface orientation, this method returns a function
            that, in turn, returns the (psi, theta) tuple of a given reflection
            and at a given energy.
        """
        if energy==None:
            if self.subs.has_key(self.energy):
                self.subs.pop(self.energy)
        else:
            self.subs[self.energy] = energy
        coordfunc = {}
        for miller in itertools.product([0,"h"], [0,"k"], [0,"l"]):
            key = "".join(map(str, miller))
            smth = zip(("h", "k", "l"), miller)
            kwargs = dict(filter(lambda x:x[1]==0, smth))
            kwvar = tuple(sorted(dict(filter(lambda x:x[1]!=0, smth)).keys()))
            theta, psi = self.calc_reflection_angles(orientation, **kwargs)
            print key
            coordfunc[key]  = lambdify(("psi_2", "epsilon") + kwvar, [psi.subs(self.subs), theta.subs(self.subs)], "numpy")
        def coordinates(miller, energy, psi2):
            key = zip(("h", "k", "l"), miller)
            kwargs = dict(filter(lambda x:x[1]!=0, key))
            key = map(lambda x: x[0] if x[1]!= 0 else "0", key)
            key = "".join(key)
            return coordfunc[key](psi2, energy, **kwargs)
        
        return coordinates

