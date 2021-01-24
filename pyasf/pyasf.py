"""
    Module provides symbolical calculation of the anisotropic Structure Factor
    of a Crystal of given Space Group and Asymmetric Unit up to
    dipole-quadrupole (DQ) approximation.

    Written by Carsten Richter (carsten.richter@desy.de)
    http://www.desy.de/~crichter
"""
from __future__ import print_function
import itertools
import copy
import six
import sympy as sp
import numpy as np
import difflib
import collections
import CifFile

if six.PY2:
    from StringIO import StringIO
    from urllib2 import urlopen
    ifilter = itertools.ifilter
else:
    from io import StringIO
    from urllib.request import urlopen
    ifilter = filter

from time import time
from fractions import Fraction
from scipy.interpolate import interp1d
from scipy import ndimage
from sympy.utilities import lambdify
from . import elements
from .functions import *
from . import FormFactors
#mydict = dict({"Abs":abs})
#epsilon=1.e-10

import rexs.xray.interactions as xi


"""
    TODO:
        DQ part imaginary?
"""




def translate_fdmnes(names):
    """
        Function that renames tensor components from
        FDMNES to the internal notation of pyasf.

        E.g. I_xyz -> f_dq_z12
    """
    indices = dict(x=1, y=2, z=3)
    for i,name in enumerate(names):
        sym, ind = name.split("_")
        if sym=="D":
            sym = "dd"
            ind = "%i%i"%(indices[ind[0]], indices[ind[1]])
        elif sym=="I":
            sym = "dq"
            ind = "%s%i%i"%(ind[2], indices[ind[0]], indices[ind[1]])
        else:
            continue
        names[i] = "%s_%s"%(sym, ind)

    return names



# Future:
#class Atom(object):
#    def __init__(self, label, isotropic=True, edge_shift=0, occupancy=1, charge=None, assume_complex=True):
#        self.label = label
#        self.charge = charge
#        self._isotropic = isotropic


class _named2darray(np.ndarray):
    def set_fields(self, fields):
        if len(fields) != self.shape[1]:
            raise ValueError("shape mismatch: number of columns and length of "
                             "fields must agree")
        if not all(map(lambda s: isinstance(s, six.string_types), fields)):
            raise ValueError("Fields must be sequence of type str")

        self._fields = fields

    def __getitem__(self, key):
        if hasattr(self, "_fields") and key in self._fields:
            key = Ellipsis, self._fields.index(key)
        elif hasattr(self, "_fields") and hasattr(key, "__iter__"):
            if all([k in self._fields for k in key]):
                key = Ellipsis, [self._fields.index(k) for k in key]
        return super(_named2darray, self).__getitem__(key)

    def __setitem__(self, key, val):
        if hasattr(self, "_fields") and key in self._fields:
            key = Ellipsis, self._fields.index(key)
        elif hasattr(self, "_fields") and hasattr(key, "__iter__"):
            if all([k in self._fields for k in key]):
                key = Ellipsis, [self._fields.index(k) for k in key]
        return super(_named2darray, self).__setitem__(key, val)

def named2darray(array, fields):
    assert array.ndim == 2, "Number of dimensions should be 2"
    assert array.shape[1] == len(fields), \
            "Number of columns and fields do not match"
    array = _named2darray(array.shape,
                          dtype = array.dtype,
                          buffer = array)
    array.set_fields(fields)
    return array


class SymbolDict(dict):
    """
        Dictionary class that translates Symbol names as key
        input to Symbol objects.
    """
    def __getitem__(self, sym):
        #if not self.has_key(sym) and isinstance(sym, str):
        if not sym in self and isinstance(sym, six.string_types):
            syms = list(self.keys())
            syms_str = [S.name for S in syms if isinstance(S, sp.Symbol)]
            num = syms_str.count(sym)
            if num==0:
                raise KeyError("Symbol not found: `%s`"%sym)
            elif num > 1:
                raise KeyError("Multiple symbols found for name: `%s`"%sym)
            else:
                sym = syms[syms_str.index(sym)]
        return super(SymbolDict, self).__getitem__(sym)


class AttrDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __dir__(self):
        return super(AttrDict, self).__dir__() + list(self)



def mkfloat(string, get_digits=False):
    """
        Simple parser for numeric values in .cif files
    """
    if isinstance(string, (six.integer_types, float)):
        return float(string)
    assert isinstance(string, six.string_types), "Invalid input. Need str."
    string = string.strip()
    string = string.replace(",", ".")
    i = string.find("(")
    if i>=0:
        string = string[:i]
    digits = len(string.split(".")[1]) if "." in string else 0
    try:
        res = float(string)
    except:
        res = 0
    if get_digits:
        return res, digits
    else:
        return res

def applymethod(Arr, Method, *args):
    """
        Function that applies a method inherent to each array
        element to all array elements.

        Inputs:
            Arr : numpy.ndarray
            Method : str -- the method

            all further arguments will be passed on to the method
    """
    for ele in np.nditer(Arr, flags=["refs_ok"], op_flags=['readwrite']):
        if hasattr(ele.item(), Method):
            ele[...] = getattr(ele.item(), Method)(*args)

def applyfunc(Arr, Func):
    """
        Function that applies a function to each array
        element

        Inputs:
            Arr : numpy.ndarray
            Func : the function
    """
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
            - reduce degrees of freedom in their tensor quantities (ADPs, form factors)
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
    eps = 100*np.finfo(np.float64).eps
    u = 1.660538921e-27 # atomic mass unit
    electron_radius = 2.8179403e-15 # meters
    avogadro = 6.022142e23
    eV_A = 12398.42
    boltzmann = 1.380658e-23
    hbar = 1.054571628e-34

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
                                - pair (space group number, space group symbol)
                                  to specify setting
                            or
                                - path to .cif-file
        """
        self.sg_sym = None
        self.AU_positions = {} # only asymmetric unit
        self.AU_formfactors = {} # only asymmetric unit, isotropic
        self.AU_formfactorsDD = {} # only asymmetric unit, pure dipolar
        self.AU_formfactorsDQ = {} # only asymmetric unit, dipolar-quadrupolar interference
        self.AU_formfactorsDDc = {} # only asymmetric unit, pure dipolar, cartesian
        self.AU_formfactorsDQc = {} # only asymmetric unit, dipolar-quadrupolar interference, cartesian
        self.miller = sp.symbols("h k l", integer=True)
        self.subs = SymbolDict()
        self.energy = sp.Symbol("epsilon", real=True, positive=True)
        self.subs.update(dict(zip(self.miller, self.miller)))
        self.S = dict([(s.name, s) for s in self.miller]) # dictionary of all symbols
        self.S["q"] = sp.Symbol("q", real=True, positive=True)
        self.elements = {}
        self.Uiso = collections.defaultdict(float)
        self.Uaniso = {}
        self.U = dict() # dictionary of anisotropic mean square displacement
        self.dE = dict() # dictionary of edge shifts
        self.f0func = FormFactors.FormFactors(kwargs.pop("f0table", "ITC"))
        self.feff_func = dict()
        self.f = dict()
        self.f0 = dict()
        self.occupancy = dict()
        self.charges = collections.defaultdict(int)
        self.masses = dict()
        self.debye_temperature = dict()
        self.einstein_temperature = dict()
        self.vectors = AttrDict()

        if str(structure).isdigit():
            structure = int(structure)
            if structure<=230:
                print("Setting up space group %i..."%structure)
                self._init_lattice(int(structure))
            else:
                print("Looking up Crystallography Open Database for entry %i..."%structure)
                codurl = "http://www.crystallography.net/cod/%i.cif"%structure
                handle = urlopen(codurl)
                text = handle.read()
                handle.close()
                ciffile = StringIO(text.decode())
                self.load_cif(ciffile, resonant, **kwargs)

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
        self.generators = get_generators(self.sg_num, self.sg_sym) # fetch the space group generators
        if self.sg_sym!=None:
            self.transform = transform = get_ITA_settings(self.sg_num)[self.sg_sym]
        metrik = get_cell_parameters(self.sg_num, self.sg_sym)
        self._metrik = metrik
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.system = metrik

        recparam = get_rec_cell_parameters(*metrik[0:6])
        self.ar, self.br, self.cr = recparam[:3] # reciprocal lattice parameters
        self.alphar, self.betar, self.gammar = recparam[3:6] # reciprocal lattice angles
        self.M, self.Minv = recparam[6:8] # transformation matrices
        dsub = {self.a:1, self.b:1, self.c:1}
        self.M0 = self.M.subs(dsub)#.subs(self.subs)
        self.M0inv = self.Minv.subs(dsub)#.subs(self.subs)
        self.metric_tensor, self.metric_tensor_inv = recparam[8:10] # metric tensors
        self.metric_tensor_0 = self.metric_tensor.subs(dsub)#.subs(self.subs)
        self.metric_tensor_inv_0 = self.metric_tensor_inv.subs(dsub)#.subs(self.subs)
        self.G = sp.Matrix(self.miller)
        # International Tables for Crystallography (2006). Vol. D, ch. 1.1, pp. 3-33
        # G is a vector in reciprocal space -> dual space, covariant
        # To transform it to cartesian space:
        # Gc = M * metric_tensor_inv * G
        #    = Minv.T * G
        self.Gc = self.Minv.T * self.G
        self.Gc.simplify()
        self.q = sp.sqrt(self.Gc.dot(self.Gc))
        self.qfunc = makefunc(self.q, sp)
        self.V = sp.sqrt(self.metric_tensor.det())

    def add_atom(self, label,
                       position,
                       isotropic=True,
                       assume_complex=True,
                       dE=0,
                       occupancy=1,
                       charge=None,
                       max_denominator=1000):
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

        if not isinstance(label, six.string_types):
            raise TypeError("Invalid label. Need string.")
        if len(position) != 3:
            raise TypeError("Enter 3D position object!")
        position = list(position)
        for i in range(3):
            if isinstance(position[i], six.string_types):
                position[i] = sp.S(Fraction(position[i]).limit_denominator(max_denominator))
            else:
                pos = Fraction("%.15g"%position[i]).limit_denominator(max_denominator)
                if (pos.denominator < max_denominator/10.):
                    position[i] = pos
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
        self.AU_positions[label] = sp.Matrix(position)
        self.dE[label] = dE
        if charge!=None:
            self.charges[label] = int(charge)

        self.occupancy[label] = occupancy

        ind = range(3)
        U = sp.zeros(3,3)
        for i,j in itertools.product(ind, ind):
            if i<=j:
                positive = True if (i==j) else None
                Sym = sp.Symbol("U_%s_%i%i"%(label, i+1, j+1), real=True, positive=positive)
                self.S[Sym.name] = Sym
                U[i,j] = U[j,i] = Sym
        self.U[label] = U

        ### FORM FACTORS:
        if assume_complex:
            my_ff_args = dict({"complex":True})
        else:
            my_ff_args = dict({"real":True})
        if isotropic:
            Sym = sp.Symbol("f_" + element, **my_ff_args)
            if Sym.name in self.S:
                Sym = self.S[Sym.name]
            else:
                self.S[Sym.name] = Sym
            self.AU_formfactors[label] = Sym
        else:
            self.resonant = element
            Sym = sp.Symbol("f_" + label + "_0", real=True, positive=True)
            self.S[Sym.name] = Sym
            self.AU_formfactors[label] = Sym
            f_DD = np.zeros((3,3), dtype=object)
            for i,j in itertools.product(ind, ind):
                if i<=j:
                    Sym = sp.Symbol("f_%s_dd_%i%i"%(label, i+1, j+1), **my_ff_args)
                    self.S[Sym.name] = Sym
                    f_DD[i,j] = Sym
                    if i<j:
                        f_DD[j,i] = Sym
            applymethod(f_DD, "simplify")

            ######### Transformation to lattice units:
            # M transforms vector from crystal units in direct space
            # to the cartesian system. Minv does the opposite. => M.T transforms cartesian tensor to direct tensor
            # M0 transforms the tensor from cartesian system to
            # crystal units allowing us to apply symmetry from ITC:
            self.AU_formfactorsDD[label] = full_transform(self.M0, f_DD) # full transform needs the transposed Matrix as input (unfortunately)
            self.AU_formfactorsDDc[label] = f_DD
            # transformation to lattice units:
            # self.AU_formfactorsDD[label] = full_transform(self.B_0.inv().T, f_DD)

            f_DQ = np.zeros((3,3,3), dtype=object)
            kindices = ("x", "y", "z")
            for h,i,j in itertools.product(ind, ind, ind):
                if j>=h:
                    Sym = sp.Symbol("f_%s_dq_%s%i%i"%(label, kindices[h], i+1, j+1), **my_ff_args)
                    self.S[Sym.name] = Sym
                    f_DQ[h,i,j] = Sym
                    if j>h:
                        f_DQ[j,i,h] = Sym
                        # See eq. 12 in Kokubun et al. http://dx.doi.org/10.1103/PhysRevB.82.205206
            applymethod(f_DQ, "simplify")
            # transformation to lattice units:
            self.AU_formfactorsDQ[label]  = full_transform(self.M0, f_DQ)
            self.AU_formfactorsDQc[label] = f_DQ
            #self.AU_formfactorsDQ[label] = full_transform(self.B_0, f_DQ)
            # transformation to lattice units:
            #self.AU_formfactorsDQ[label] = full_transform(self.B_0.inv().T, f_DQ)
            if self.DEBUG:
                print(self.AU_formfactorsDD[label], self.AU_formfactorsDQ[label])


    def get_ion(self, label):
        charge = self.charges[label]
        ion = self.elements[label]
        if charge > 0:
            ion += "%i+"%abs(charge)
        elif charge < 0:
            ion += "%i-"%abs(charge)
        else:
            pass
        return ion

    def load_cif(self, fname, resonant="", max_denominator=100, blocknr=0, digits=4):
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
            self.cif = cif = cf.items()[blocknr][1]
        except Exception as e:
            print("File doesn't seem to be a valid .cif file: %s"%fname)
            raise IOError(e)
        if cif.has_key("_symmetry_int_tables_number"):
            sg_num = int(cif["_symmetry_int_tables_number"])
        elif cif.has_key("_space_group_IT_number"):
            sg_num = int(cif["_space_group_IT_number"])
        else:
            sg_num = None
            sg_sym = cif["_symmetry_space_group_name_h-m"]
            sg_sym = "".join(sg_sym.split())
            for i in range(1, 230):
                sett = get_ITA_settings(i)
                if sg_sym in sett:
                    sg_num = i
        if sg_num is None:
            raise ValueError("space group number could not be determined from .cif file `%s`:"%fname)
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
                if sg_sym[-1] in ["R", "H"] and ":" not in sg_sym:
                    sg_sym = sg_sym[:-1] + ":" + sg_sym[-1]

                settings = list(ITA.keys())
                sg_sym = sg_sym.lower()
                ratios = [difflib.SequenceMatcher(a=sg_sym, b=set.lower()).ratio() for set in settings]
                setting = settings[np.argmax(ratios)]
                print("  Identified symbol `%s' from .cif entry `%s'"\
                      %(setting, sg_sym))
                self.sg_sym = sg_sym = setting


        def getcoord(cifline):
            coord, digits = mkfloat(cifline, get_digits=True)
            if max_denominator is None:
                return coord
            coordrat = Fraction(coord)
            coordrat = coordrat.limit_denominator(max_denominator)
            if abs(coord-float(coordrat)) < (1*10**-digits):
                coord = coordrat
            return coord

        def getangle(cifline):
            ang, digits = mkfloat(cif[cifline], get_digits=True)
            ang = ang/180
            if max_denominator is None:
                return coord
            angrat = Fraction("%.15f"%ang)
            angrat = angrat.limit_denominator(max_denominator)
            if abs(ang-float(angrat)) < (1*10**-digits/180.):
                ang = angrat
            return sp.S(ang) * sp.pi

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
        for key in loop.keys():
            loop[key.lower()] = loop[key]
        for line in loop:
            label = line._atom_site_label
            #symbol = line._atom_site_type_symbol
            try:
                charge = int(line._atom_site_type_symbol[-2:][::-1])
            except:
                charge = 0
            symbol = label[:2].capitalize()
            symbol = symbol if symbol in elements.Z else symbol[0]
            px = sp.S(getcoord(line._atom_site_fract_x))
            py = sp.S(getcoord(line._atom_site_fract_y))
            pz = sp.S(getcoord(line._atom_site_fract_z))
            occ = mkfloat(line._atom_site_occupancy) if loop.has_key("_atom_site_occupancy") else 1
            position = [stay_in_UC(p) for p in (px, py, pz)]
            isotropic = (symbol not in resonant)
            if loop.has_key("_atom_site_u_iso_or_equiv"):
                iso = mkfloat(line._atom_site_u_iso_or_equiv)
            elif loop.has_key("_atom_site_b_iso_or_equiv"):
                iso = mkfloat(line._atom_site_b_iso_or_equiv)
                iso /= 8. * np.pi**2
            else:
                iso = 0

            self.Uiso[label] = iso
            self.add_atom(label, position, isotropic, assume_complex=True,
                          occupancy=occ, charge=charge, max_denominator=max_denominator)
            self.eps = 10**(-digits)

        vr = self.ar, self.br, self.cr
        if cif.has_key("_atom_site_aniso_label"):
            loop = cif.GetLoop("_atom_site_aniso_label")
            for key in loop.keys():
                loop[key.lower()] = loop[key]
            for num, line in enumerate(loop):
                label = line._atom_site_aniso_label
                self.Uaniso[label] = Uaniso = sp.zeros(3,3)
                for (i,j) in [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]:
                    kw_U = "_atom_site_aniso_u_%i%i"%(i,j)
                    kw_B = "_atom_site_aniso_b_%i%i"%(i,j)
                    kw_beta = "_atom_site_aniso_beta_%i%i"%(i,j)
                    if loop.has_key(kw_U):
                        value = mkfloat(loop[kw_U][num])
                    elif loop.has_key(kw_B):
                        value = mkfloat(loop[kw_B][num])
                        value /= 8. * np.pi**2
                    elif loop.has_key(kw_beta):
                        value = mkfloat(loop[kw_beta][num])
                        value /= 2*np.pi**2 * vr[i-1] * vr[j-1] # http://dx.doi.org/10.1107/S0108767396005697
                    Uaniso[i-1,j-1] = Uaniso[j-1,i-1] = value


    def get_tensor_symmetry(self, labels = None):
        """
            Applies Site Symmetries of the Space Group to the Tensors.
            (formfactors and Debye Waller U)
        """
        if labels == None:
            labels = self.U.keys()
        self._equations = {}
        self._symmetries = {}
        self._Usymmetries = {}
        self._Uequations = {}
        M_r = sp.Matrix([self.ar, self.br, self.cr])
        M_r = M_r * M_r.T
        for label in labels:
            if self.DEBUG:
                print(label)
            equations = set()
            Uequations = set()
            U = self.U[label]
            Beta = U.multiply_elementwise(M_r)
            #Beta = full_transform(self.metric_tensor, Beta) # transform to direct lattice
            site = self.AU_positions[label]
            for ii, generator in enumerate(self.generators):
                W = generator[:,:3]
                w = generator[:,3]
                #if self.DEBUG: print w, W
                new_position = W * site + w
                new_position = new_position.applyfunc(stay_in_UC)
                diff = (new_position - site)
                dist = []

                # check if atom was moved to neighboring cell:
                for v in itertools.product((-1,0,1), (-1,0,1), (-1,0,1)):
                    v = sp.Matrix(v)
                    diff_ = diff + v
                    if diff_.has(sp.Symbol):
                        # then it's not the same position
                        dist.append(1)
                        continue
                    dist.append(diff_.dot(diff_))

                dist = min(dist) if dist else 0

                if self.DEBUG:
                    print("Generator #%i"%ii)
                    print((dist, diff))
                #if (new_position == self.AU_positions[label]).all(): #1.8.13
                if dist < self.eps:
                    if self.DEBUG:
                        print(generator)
                    # International Tables for Crystallography (2006). Vol. D, ch. 1.1, pp. 3-33
                    # incident polarization is contravariant => f is 2 times covariant
                    # -> first axis in DD
                    # -> second axis in DQ
                    # W rotates position to new position
                    # -> that is the same as rotating polarization by W.inv
                    # -> To get representation of rotated tensor:
                    #     - rotate contracting Vectors (polarization) in opposite W.inv
                    #     ==> Transform Tensor with Rotation W

                    if label in self.AU_formfactorsDD:
                        fDD = self.AU_formfactorsDD[label]
                        new_DD = full_transform(W.inv(), fDD)
                        #new_DD = full_transform(W, fDD)
                        equations.update((fDD - new_DD).ravel())

                    if label in self.AU_formfactorsDQ:
                        fDQ = self.AU_formfactorsDQ[label]
                        new_DQ = full_transform(W.inv(), fDQ)
                        #new_DQ = full_transform(W, fDQ)
                        equations.update((fDQ - new_DQ).ravel())

                    # U is 2 times contravariant so they transform the same
                    #  way as the basis
                    #  Therefore inverse
                    #print W, W.inv().T
                    #new_U =  full_transform(W.inv(), U)
                    #Uequations.update(np.ravel(new_U - U))
                    new_Beta = full_transform(W.T, Beta)
                    #new_Beta = full_transform(W.inv().T, Beta)
                    if self.DEBUG:
                        sp.pprint([sp.Matrix(new_Beta),Beta])
                    Uequations.update(np.ravel(new_Beta - Beta))

            equations.discard(0)
            Uequations.discard(0)
            self._equations[label] = equations
            self._Uequations[label] = Uequations


            if equations:
                ffSym = set()
                ffSym.update(np.ravel(self.AU_formfactorsDDc.get(label, 0)))
                ffSym.update(np.ravel(self.AU_formfactorsDQc.get(label, 0)))
                ffSym = [s for s in ffSym if s.is_Symbol]
                #ffSym = [Sym S in self.S.iteritems() if k.startswith("f_")]
                symmetries =  sp.solve( equations, ffSym, dict=True, manual=True)
                self._symmetries[label] = symmetries = symmetries[0]
                if self.DEBUG:
                    print(symmetries)
                applymethod(self.AU_formfactorsDD[label], "subs", symmetries)
                applymethod(self.AU_formfactorsDD[label], "simplify")
                applymethod(self.AU_formfactorsDQ[label], "subs", symmetries)
                applymethod(self.AU_formfactorsDQ[label], "simplify")
                self.AU_formfactorsDDc[label]  = full_transform(self.M0inv, self.AU_formfactorsDD[label])
                self.AU_formfactorsDQc[label]  = full_transform(self.M0inv, self.AU_formfactorsDQ[label])
                applymethod(self.AU_formfactorsDDc[label], "simplify")
                applymethod(self.AU_formfactorsDQc[label], "simplify")

            Usymmetries = sp.solve(Uequations, sp.flatten(U), dict=True, manual=True)
            if Usymmetries:
                self._Usymmetries[label] = Usymmetries = Usymmetries[0]
                self.U[label] = U.subs(Usymmetries)
                self.U[label].simplify()

            if self.DEBUG:
                if label in self.AU_formfactorsDD:
                    print(self.AU_formfactorsDD[label])
                if label in self.AU_formfactorsDQ:
                    self.AU_formfactorsDQ[label]

    def U_cartesian(self, label, symbolic=True):
        """
            Calculate atomic displacement parameter U in cartesian coordinates
            according to http://dx.doi.org/10.1107/S0108767396005697
        """
        if symbolic:
            U = self.U[label]
        else:
            U = self.Uaniso[label]
        Dij = self.M * sp.diag(self.ar, self.br, self.cr)
        Uc = sp.Matrix(full_transform(Dij, U))

        if not symbolic:
            Uc = Uc.subs(self.subs)
        return Uc

    def get_thermal_ellipsoids(self, label):
        """
            Returns eigenvalues and eigenvectors for the atomic displacement
            parameter in cartesian coordinates U_ij^c according to
            http://dx.doi.org/10.1107/S0108767396005697
        """

        Uc = self.U_cartesian(label, False)
        Uc = np.array(Uc).astype(float)
        Ueq = Uc.trace()/3.

        eigval, eigvec = np.linalg.eigh(Uc)
        #iso = eigval.mean()
        #aniso = eigval.max() / eigval.min()

        #direction_c = eigvec[:,eigval.argmax()]
        #direction = self.Minv.dot(direction_c)
        #direction = direction.subs(self.subs).n()

        return eigval, eigvec

    def build_unit_cell(self):
        """
            Generates all Atoms (orbit) of the Unit Cell with their
            corresponding tensor quantities
        """
        self.multiplicity = collections.defaultdict(int)
        self.positions = collections.defaultdict(list)
        #self.formfactors = collections.defaultdict(list)
        self.formfactorsDD = collections.defaultdict(list)
        self.formfactorsDQ = collections.defaultdict(list)
        self.Beta = collections.defaultdict(list)
        M_r = sp.Matrix([self.ar, self.br, self.cr])
        M_r = M_r * M_r.T
        self._positions = []
        self._labels = []
        for label, site in self.AU_positions.items():
            U = self.U[label]
            Beta = 2 * sp.pi**2 * U.multiply_elementwise(M_r)
            for generator in self.generators:
                W = generator[:,:3]
                w = generator[:,3]
                #print W, self.AU_positions[label], w
                new_position = W * site + w
                new_position = new_position.applyfunc(stay_in_UC)
                dist = []

                # check if atom was moved to neighboring cell:
                for position in self.positions[label]:
                    diff = new_position - position
                    for v in itertools.product((-1,0,1), (-1,0,1), (-1,0,1)):
                        v = sp.Matrix(v)
                        diff_ = diff + v
                        if diff_.has(sp.Symbol):
                            # then it's not the same position
                            continue
                        dist.append(diff_.dot(diff_))

                #dist = [d for d in dist if np.isscalar(d) or not hassymb(d)]

                if not dist or not min(dist) < self.eps:
                    #if new_position not in self.positions[label]:
                    self.Beta[label].append(sp.Matrix(full_transform(W.T, Beta)))
                    #self.Beta[label].append(sp.Matrix(full_transform(W.inv().T, Beta)))
                    #if label in self.AU_formfactors:
                    #    self.formfactors[label].append(self.AU_formfactors[label])
                    if label in self.AU_formfactorsDD:
                        #self.formfactorsDD[label].append(full_transform(W, self.AU_formfactorsDD[label]))
                        self.formfactorsDD[label].append(full_transform(W.inv(), self.AU_formfactorsDD[label]))
                    if label in self.AU_formfactorsDQ:
                        #self.formfactorsDQ[label].append(full_transform(W, self.AU_formfactorsDQ[label]))
                        self.formfactorsDQ[label].append(full_transform(W.inv(), self.AU_formfactorsDQ[label]))
                    self.positions[label].append(new_position)
                    #sp.pprint(self.Beta[label][-1])
                    #sp.pprint(new_position)
                    self._positions.append(new_position.T)
                    self._labels.append(label)
                    self.multiplicity[label] += 1
        # rough estimate of the infimum of the distance of atoms:
        self._positions = np.vstack(self._positions)
        self._labels = np.array(self._labels)
        self.numato = sum(self.multiplicity.values())
        self.mindist = 1./self.numato**(1./3)/1000.




    def transform_unit_cell_positions(self, generator):
        """
            Generates all Atoms of the Unit Cell.
        """
        self.multiplicity = collections.defaultdict(int)

        _positions = []
        _labels = []
        for label, position in zip(self._labels, self._positions):
            W = generator[:,:3]
            w = generator[:,3]
            new_position = W * position + w
            new_position = new_position.applyfunc(stay_in_UC)
            _positions.append(np.ravel(new_position))
            _labels.append(label)
        return _labels, _positions




    def get_nearest_neighbors(self, label, num=1):
        num = int(num)
        if not num > 0:
            return

        pos = np.array(self.AU_positions[label], dtype=float).ravel()
        diff = (pos - self._positions.astype(float))%1
        diff = np.vstack([diff, diff-np.array((0,0,1))])
        diff = np.vstack([diff, diff-np.array((0,1,0))])
        diff = np.vstack([diff, diff-np.array((1,0,0))])

        #diff =  (diff+0.5)%1-0.5
        M = np.array(self.M.subs(self.subs).n()).astype(float)
        self._diff = diff = diff.dot(M.T)
        #self._dist = dist = np.sqrt((diff**2).sum(1))
        self._dist = dist = np.linalg.norm(diff, axis=1)
        ind = dist.argsort()
        indlbl = ind%len(self._labels)
        return self._labels[indlbl[1:num+1]], dist[ind[1:num+1]], diff[ind[1:num+1]]


    def get_density(self):
        """
            Calculates the density in g/cm^3 from the structure.
        """
        assert hasattr(self, "positions"), \
            "Unable to find atom positions. Did you forget to perform the unit_cell.build_unit_cell() method?"
        self.species = np.unique(list(self.elements.values()))
        self.stoichiometry = collections.defaultdict(float)
        for label in self.elements:
            self.stoichiometry[self.elements[label]] += self.multiplicity[label] * self.occupancy[label]

        self.weights = dict([(atom, xi.get_element(atom)[1]) for atom in self.species])
        div = min(self.stoichiometry.values())
        components = ["%s%.2g"%(item[0], item[1]/div) for item in self.stoichiometry.items()]
        components = map(lambda x: x[:-1] if (x[-1]=="1" and not x[-2].isdigit()) else x, components)
        self.SumFormula = "".join(components)


        self.total_weight = sum([self.weights[atom]*self.stoichiometry[atom] for atom in self.species])
        self.density = self.u * self.total_weight/1000. / (self.V*1e-10**3) # density in g/cm^3
        return float(self.density.subs(self.subs).n())

    def get_stoichiometry(self):
        """
            Returns the stoichiometry of the structure
        """
        self.get_density()
        return self.SumFormula

    def calc_structure_factor(self, miller=None, DD=True, DQ=True, Temp=True, simplify=False,
                                    subs=False, evaluate=False, Uaniso=True):
        """
            Takes the three Miller-indices of Type int and calculates the
            Structure Factor in reciprocal basis.

            Inputs:
                miller : 3 tuple(int)
                    The h,k,l of the Bragg reflection. If `None`, structure factor of
                    all h,k,l is computed.

                DD, DQ : bool
                    Whether to calculate the dipole-dipole or dipole-quadrupole
                    tensor components

                Temp : bool
                    Whether to take into account thermal vibrations. Use a float
                    to lazily scale the effect of temperature.

                simplify : bool
                    If True, try to simplify the scalar part of the structure
                    factor. A shorter expression can lead to enhanced speed
                    during numeric evaluation.

                subs : bool, sequence of symbols
                    Whether to insert numeric values for symbols using
                    the `unit_cell.subs` dictionary. If input is a sequence
                    of `sympy.Symbol`, only these symbols will be replaced.

                evaluate : bool
                    If True, simplify expressions by numerical evaluation
                    where possible

                Uaniso : bool
                    Whether to take into account anisotropic thermal
                    displacements.

            Outputs:
                F : symbolic expression of the structure factor

        """
        if miller is None:
            miller = self.miller
        self.hkl(*miller)
        if not hasattr(self, "positions"):
            self.build_unit_cell()
        G = self.G.subs(self.subs)
        Gc = self.Gc.subs(self.subs)


        self.q = self.qfunc.dictcall(self.subs)
        self.d = 1/self.q
        self.theta = sp.asin(self.eV_A/(2*self.d*self.energy))

        self.F_0 = sp.S(0)
        DD = DD and len(self.formfactorsDD)>0
        DD = DQ and len(self.formfactorsDQ)>0
        if DD:
            self.F_DD = np.zeros((3,3), object)
        if DQ:
            self.F_DQin = np.zeros((3,3,3), object)
            self.F_DQsc = np.zeros((3,3,3), object)
        if isinstance(Temp, bool):
            Temp = int(Temp)

        for label in self.positions: # get position, formfactor and symmetry if DQ tensor
            o = self.occupancy[label]
            if Uaniso and label in self.Uaniso:
                if subs is True:
                    Uval = self.Uaniso[label]
                    vald = zip(sp.flatten(self.U[label]), sp.flatten(Uval))
                    for (k,v) in vald:
                        if isinstance(k, sp.Symbol):
                            self.subs[k] = v
            else:
                Uval = self.Uiso[label]

            F_0_orbit = 0
            for i, r in enumerate(self.positions[label]):
                if Temp:
                    if Uaniso:
                        # International Tables for Crystallography (2006).
                        # Vol. D, Chapter 1.9, pp. 228.242.
                        Beta = self.Beta[label][i]
                        DW = sp.exp(-G.dot(Beta*G)*Temp)
                    else:
                        DW = sp.exp(-2 * sp.pi**2 * self.q**2 * Uval * Temp)
                else:
                    DW = 1

                phase = sp.exp(2*sp.pi*sp.I * G.dot(r))
                if self.DEBUG:
                    print((r, G.dot(r)))

                F_0_orbit += phase * DW

                if DD and label in self.formfactorsDD:
                    f = self.formfactorsDD[label][i]
                    self.F_DD += o * f * phase * DW

                if DQ and label in self.formfactorsDQ:
                    f_in = self.formfactorsDQ[label][i]
                    f_sc = - f_in.copy().transpose(0,2,1)
                    #applymethod(f_sc, "conjugate") # only real parameters as in FDMNES!!!!?
                    #self.F_DQ += o * (f_in - f_out) * phase * DW
                    self.F_DQin += o * f_in * phase * DW
                    self.F_DQsc += o * f_sc * phase * DW

            self.F_0 += o * self.AU_formfactors[label] * F_0_orbit

        if subs is True:
            self.F_0 = self.F_0.subs(self.subs)
        elif isinstance(subs, collections.Sequence):
            subitems = [(k, self.subs[k]) for k in subs]
            self.F_0 = self.F_0.subs(subitems)

        if evaluate:
            self.F_0 = self.F_0.n().expand()
        if simplify:
            self.F_0 = self.F_0.simplify()

        self.F_0_func = makefunc(self.F_0)
        # if self.F_0.has(sp.Symbol):
        #     self.F_0_func = makefunc(self.F_0)
        # else:
        #     self.F_0_func = None

        # simplify:

        if DD:
            if subs:
                applymethod(self.F_DD, "subs", self.subs)
            if evaluate:
                applymethod(self.F_DD, "n")
            self.F_DD = ArraySimp2(self.F_DD)
        if DQ:
            if subs:
                applymethod(self.F_DQin, "subs", self.subs)
                applymethod(self.F_DQsc, "subs", self.subs)
            if evaluate:
                applymethod(self.F_DQin, "n")
                applymethod(self.F_DQsc, "n")
            self.F_DQin = ArraySimp2(self.F_DQin)
            self.F_DQsc = ArraySimp2(self.F_DQsc)

        return self.F_0

    def hkl(self, *miller, **kw):
        """
            Set or get the current Miller indices hkl.
        """
        if miller:
            self.subs.update(zip(self.miller, miller))
        elif kw:
            miller = map(lambda S: S.name, self.miller)
            miller = dict(zip(miller, self.miller))
            subs = [(miller[k], v) for k,v in kw.items() if k in miller]
            self.subs.update(subs)
        else:
            return tuple(self.subs[ind] for ind in self.miller)


    def transform_structure_factor(self, AAS = True, simplify=True,
                                                         subs=True):
        """

            First transforms F to a real space, cartesian, crystal-fixed
            system ->Fc. Then transforms Fc to the diffractometer system,
            which is G along xd and sigma along zd ->Fd.

            This happens according to the work reported in:
            Acta Cryst. (1991). A47, 180-195
            [doi:10.1107/S010876739001159X]
        """
        if not hasattr(self, "F_0") or \
           (AAS and (not hasattr(self, "F_DD") or not hasattr(self, "F_DQin"))):
            raise ValueError("No Reflection initialized. "
                             " Run self.calc_structure_factor() first.")
        # now the structure factors in a cartesian system follow
        if simplify:
            self.Fc_0 = self.F_0.n().simplify()
        else:
            self.Fc_0 = self.F_0
        if AAS:
            #self.Fc_DD = full_transform(B_0_inv, self.F_DD) # RLU
            #self.Fc_DQ = full_transform(B_0_inv, self.F_DQ) # RLU
            self.Fc_DD   = full_transform(self.M0inv, self.F_DD)
            self.Fc_DQin = full_transform(self.M0inv, self.F_DQin)
            self.Fc_DQsc = full_transform(self.M0inv, self.F_DQsc)

        # and now: the rotation into the diffractometer system
        #       (means rotation G into xd-direction)
        # calculate corresponding angles

        if subs:
            Gc = self.Gc.subs(self.subs)
        else:
            Gc = self.Gc.subs(zip(self.miller, self.hkl()))

        pi = sp.pi
        if Gc[1] == 0 and Gc[0] == 0: phi = 0
        elif Gc[1] == 0 and sp.sign(Gc[0])>0: phi = 0
        elif Gc[1] == 0 and sp.sign(Gc[0])<0: phi = pi
        elif Gc[0] == 0: phi = sp.sign(Gc[1])*pi/2
        else:
            phi = sp.atan(Gc[1]/Gc[0])
            if sp.sign(Gc[0])<0:
                phi = phi + pi

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
        self.Q = self.Xi.dot(self.Phi)
        #self.Q = self.Phi.dot(self.Xi)
        #self.Q = np.dot(self.Phi, self.Xi)
        if subs:
            applymethod(self.Q, "subs", self.subs)
            #applymethod(self.Q, "n")
        if simplify:
            applymethod(self.Q, "simplify")
            if AAS:
                self.Fc_DD = ArraySimp1(self.Fc_DD)
                self.Fc_DQin = ArraySimp1(self.Fc_DQin)
                self.Fc_DQsc = ArraySimp1(self.Fc_DQsc)

        self.Fd_0 = self.Fc_0
        if AAS:
            self.Fd_DD = full_transform(self.Q.T, self.Fc_DD)
            self.Fd_DQin = full_transform(self.Q.T, self.Fc_DQin)
            self.Fd_DQsc = full_transform(self.Q.T, self.Fc_DQsc)
        self.Q = sp.Matrix(self.Q)
        self.Gd = self.Q * Gc



    def transform_rec_lat_vec(self, miller, psi=0, inv=False):
        """
            Transforms a reciprocal lattice vector to a laboratory 
            coordinate system as defined in: https://doi.org/10.1107/S0108767391011509
        """
        assert len(miller)==3, "Input has to be vector of length 3."
        miller = sp.Matrix(miller)
        if not hasattr(self, "Q"):
            self.transform_structure_factor()
        UB = self.Q * self.M #* self.metric_tensor
        if psi!=0:
            Psi = np.array([[1, 0, 0],
                           [0,sp.cos(psi),sp.sin(psi)],
                           [0, -sp.sin(psi), sp.cos(psi)]])
            UB = sp.Matrix(Psi) * UB
        if inv:
            UB = UB.inv()
        return UB * miller


    def theta_degrees(self, energy=None, h=None, k=None, l=None, rad=False):
        """
            Returns the Bragg angle (theta) in degree for a given energy in eV.
        """
        subs = zip(self.miller, (h,k,l))
        subs = filter(lambda x: x[1]!=None, subs)
        subs = dict(subs)
        if energy!=None:
            subs[self.energy] = energy
        fac = 180/sp.pi if not rad else 1
        return sp.N(self.theta.subs(subs).subs(self.subs) * fac)


    def calc_scattered_amplitude(self, psi=None, assume_imag=False, assume_real=False,
                                       DD=True, DQ=True, simplify=True, subs=True):

        self.transform_structure_factor(AAS = (DD+DQ), subs=subs, simplify=simplify)
        if psi is None:
            self.psi = psi = sp.Symbol("psi", real=True)
            self.S[psi.name] = psi

        sigma = sp.Matrix([0,0,1])
        self.vectors.sigma = sigma
        if subs:
            theta = self.theta
            k = self.energy / self.eV_A
        else:
            theta = sp.Symbol("theta_B", real=True, positive=True)
            k = sp.Symbol("k", real=True, positive=True)
            self.S[theta.name] = theta
            self.S[k.name] = k
            self.subs[theta] = self.theta
            self.subs[k] = self.energy / self.eV_A
        self.k_plus = 2 * k * sp.cos(self.theta)
        pi_i = sp.Matrix([sp.cos(theta),  sp.sin(theta), 0])
        pi_s = sp.Matrix([sp.cos(theta), -sp.sin(theta), 0])
        self.vectors.pi_i = pi_i
        self.vectors.pi_s = pi_s

        vec_k_i = sp.Matrix([-sp.sin(theta), sp.cos(theta), 0]) #alt
        vec_k_s = sp.Matrix([ sp.sin(theta), sp.cos(theta), 0]) #alt
        self.vectors.k_i = vec_k_i
        self.vectors.k_s = vec_k_s

        # introduce rotational matrix
        self.Psi = Psi = np.array([[1,            0,           0],
                                   [0,  sp.cos(psi), sp.sin(psi)],
                                   [0, -sp.sin(psi), sp.cos(psi)]])
        self.Fd_psi_0 = self.Fd_0.simplify()

        if DD:
            self.Fd_psi_DD = full_transform(Psi.T, self.Fd_DD)
            if simplify:
                applymethod(self.Fd_psi_DD, "simplify")
        if DQ:
            self.Fd_psi_DQin = full_transform(Psi.T, self.Fd_DQin)
            self.Fd_psi_DQsc = full_transform(Psi.T, self.Fd_DQsc)
            if simplify:
                applymethod(self.Fd_psi_DQin, "simplify")
                applymethod(self.Fd_psi_DQsc, "simplify")


#        # calculate symmetric and antisymmetric DQ Tensors
#        if DQ:
#            self.Fd_psi_DQ_out = self.Fd_psi_DQ.copy().transpose(0,2,1)
##            if assume_real:
##                pass
##            elif assume_imag:
##                self.Fd_psi_DQ_out *= -1
##            else:
##                applymethod(self.Fd_psi_DQ_out, "conjugate") # only real parameters as in FDMNES!!!!?
#            self.Fd_psi_DQs = (self.Fd_psi_DQ + self.Fd_psi_DQ_out)/2.
#            self.Fd_psi_DQa = -(self.Fd_psi_DQ - self.Fd_psi_DQ_out)/2.

#
#        self.Fd = np.eye(3) * self.Fd_psi_0
#        if DD:
#            self.Fd += self.Fd_psi_DD
#        if DQ:
#            self.Fd += sp.I * self.q      * self.Fd_psi_DQs[0] \
#                     + sp.I * self.vec_k_plus * self.Fd_psi_DQa[1]
#
        # The Contraction:
#        self.Fd_psi_DQ_out = self.Fd_psi_DQ.copy().transpose(0,2,1)
#        applymethod(self.Fd_psi_DQ_out, "conjugate") # only real parameters as in FDMNES!!!!?


        self.Fd = np.eye(3) * self.Fd_psi_0

        if DD:
            self.Fd += self.Fd_psi_DD

        if DQ:
            self.Fd += sp.I * k * ( np.tensordot(vec_k_i, self.Fd_psi_DQin, axes=(0,0)).squeeze() \
                                  + np.tensordot(vec_k_s, self.Fd_psi_DQsc, axes=(0,0)).squeeze())

        self.Fd = sp.Matrix(self.Fd)
        if simplify:
            self.Fd.simplify()
        #self.Fd_alt = sp.Matrix(sp.expand(self.Fd_alt))


        self.E = {}
        self.E["ss"] = (sigma.T * self.Fd * sigma)[0]
        self.E["sp"] = (pi_s.T * self.Fd * sigma)[0]
        self.E["ps"] = (sigma.T * self.Fd * pi_i)[0]
        self.E["pp"] = (pi_s.T * self.Fd * pi_i)[0]

        return self.E



    def feed_feff(self, label, energy, fprime, fsecond, fit=True):
        """
            Input function for dispersion fine structure (f', f'')
            for a certain atom indicated by `label`.

            The `fit` argument specifies whether to scale the fine
            structure in order to fit the tabulated values.
        """

        if label not in self.elements:
            raise ValueError("Atom name not found in structure: %s"%label)
        energy = np.array(energy)
        fprime = np.array(fprime)
        fsecond = np.array(fsecond)
        assert energy.ndim == fprime.ndim == fsecond.ndim == 1, \
            "Invalid dimensionality of input energy, fprime or fsecond"
        assert energy.shape == fprime.shape == fsecond.shape, \
            "Length of input arrays energy, fprime or fsecond disagree"

        element = self.elements[label]
        Z = elements.Z[element]
        if fprime.max()/Z > 0.1:
            fprime -= Z

        self.feff_func[label] = interp1d(energy, fprime + 1j*fsecond)
        self.fit_feff = fit
        return True



    def get_f1f2_isotropic(self, energy, fwhm_ev=None,
                           table="Sasaki", feff=True, force_reload=False):
        """
            Method to fetch dispersion correction values from database.
            The data is cached in and used for interpolation in the
            following calls.

            Inputs:
                energy : ndarray 1d
                    Photon energy values in eV.
                fwhm_ev : float
                    Width for gaussian broadening of the results.
                    default: 0.1
                table : string
                    Database to take f1,f2 values from. Available databases
                    listed in `rexs.xray.interactions.supported_tables`.
                feff : bool
                    Whether to use fine structure (effective f) that
                    was previously put in via `unit_cell.feed_feff`.
                force_reload : bool
                    If true, don't use cached table data.
        """

        energy = np.array(energy, dtype=float, ndmin=1)
        isort = energy.argsort()
        emin, emax = energy[isort[[0, -1]]]
        atoms = list(self.AU_positions)

        if fwhm_ev is None:
            if hasattr(self, "_ftab"):
                fwhm_ev = self._ftab.fwhm_ev
            else:
                fwhm_ev = 0.1

        if force_reload:
            pass
        elif not hasattr(self, "_ftab"):
            force_reload = True
        elif not set(self._ftab.atoms).issuperset(atoms):
            force_reload = True
        elif emin < self._ftab.x[0] or emax > self._ftab.x[-1]:
            force_reload = True
        elif self._ftab.table != table:
            force_reload = True
        elif not np.allclose(self._ftab.fwhm_ev, fwhm_ev):
            force_reload = True


        if force_reload:
            fwhm_ev = abs(fwhm_ev)
            if table=="deltafquad": # get resonant energies
                from rexs.xray import deltaf
                newenergy = []
                for label in atoms:
                    element = self.elements[label]
                    try:
                        newenergy.append(deltaf.get_energies(
                         element, emin, emax, fwhm_ev, verbose=False)[0])
                    except:
                        pass
                newenergy = np.sort(np.hstack(newenergy))
            else:
                newenergy = np.arange(emin-25, emax+25)
            ff_list = []
            for label in atoms:
                dE = self.dE[label]
                element = self.elements[label]
                Z = elements.Z[element]
                if self.DEBUG:
                    print("Fetching resonant dispersion corrections for %s "
                          "from table `%s`."%(element, table))
                if table=="deltafquad":
                    f1 = deltaf.getfquad(element, newenergy - dE,
                                                    fwhm_ev, f1f2="f1")
                    f2 = deltaf.getfquad(element, newenergy - dE,
                                                    fwhm_ev, f1f2="f2")
                    ff_list.append(f1 + 1j*f2)
                else:
                    f1, f2 = xi.get_f1f2_from_db(element, newenergy - dE,
                                                               table=table)
                    if fwhm_ev>0:
                        fwhm_steps = fwhm_ev/(newenergy[1] - newenergy[0])
                        if self.DEBUG:
                            print(fwhm_steps)
                        f1 = ndimage.gaussian_filter1d(f1, fwhm_steps/2.355)
                        f2 = ndimage.gaussian_filter1d(f2, fwhm_steps/2.355)
                    ff_list.append(f1-Z + 1j*f2)
            self._ftab = interp1d(newenergy, ff_list, kind="linear")
            self._ftab.atoms = atoms
            self._ftab.fwhm_ev = fwhm_ev
            self._ftab.table = table

        f = self._ftab(energy)
        f =  dict(zip(self._ftab.atoms, f))

        for atom in self.feff_func:
            if not feff:
                break
            if atom in f:
                func = self.feff_func[atom]
                ind  = energy >= func.x.min()
                ind *= energy <= func.x.max()
                energy_sel = energy[ind]
                f1f2 = self.feff_func[atom](energy_sel)
                if self.fit_feff:
                    imin  = energy_sel.argmin()
                    imax = energy_sel.argmax()
                    emin = max(self._ftab.x.min(), func.x.min())
                    emax = min(self._ftab.x.max(), func.x.max())
                    if emin > emax:
                        raise ValueError("invalid energy ranges")
                    feffl, feffr = func([emin, emax])
                    iatom = self._ftab.atoms.index(atom)
                    fsl, fsr = self._ftab([emin, emax])[iatom]

                    # linear correction:
                    lin = (fsr/feffr - fsl/feffl)/(emax - emin) \
                                          * (energy_sel - emin) + fsl/feffl
                    f1f2 *= lin

                f[atom][ind] = f1f2

        self.f1f2 = f
        return f


    def get_tensors_FDMNES(self, path, label, Eedge, Emin=None, Emax=None):

        if label not in self.AU_formfactorsDD:
            raise ValueError("label %s not found in list of anisotric atoms"
                             %str(label))
        with open(path) as fh:
            header = fh.readline()
        header = header.split()
        if not header[0]=="Energy" and header[1]=="D_xxp":
            raise ValueError("%s doesn't seem to be a valid file of cartesian "
                             "tensors. (Need convoluted values)")

        data = np.loadtxt(path, skiprows=1)
        E = data[:,0] + Eedge
        ind = np.ones(len(E), dtype=bool)
        if Emin!=None:
            ind *= E > Emin
        if Emax!=None:
            ind *= E < Emax
        E = E[ind]
        data = data[ind]
        data = data.astype(complex)[:,1:]
        data[:,::2] += 1j * data[:,1::2]
        data = data[:,::2]
        header = map(lambda s: s.strip("p"), header[1::2])

        header = translate_fdmnes(header)

        data = named2darray(data.copy(), header)

        fiso = np.mean([data["dd_%s"%s] for s in ["11", "22", "33"]], 0)

        data["dd_11"] -= fiso
        data["dd_22"] -= fiso
        data["dd_33"] -= fiso

        ftab = self.get_f1f2_isotropic(E)[label]
        lin = ((ftab[-1] - fiso[-1]) - (ftab[0] - fiso[0]))/(E[-1] - E[0]) \
                          * (E - E[0]) + (ftab[0] - fiso[0])
        fiso += lin
        self.feed_feff(label, E, fiso.real, fiso.imag)

        if not hasattr(self, "f_aniso"):
            self.f_aniso = dict()
        if not hasattr(self, "f_aniso_func"):
            self.f_aniso_func = dict()

        self.f_aniso[label] = data
        self.f_aniso_func[label] = interp1d(E, data.T,
                                            bounds_error = False,
                                            fill_value = 0.)
        self.f_aniso_func[label].components = header

        return E, data



    def get_absorption_isotropic(self, energy, density=None, table="Sasaki",
                                       fwhm_ev=None, f1f2=None):
        """
            Returns the linear *photoelectric* absorption coefficient of the
            structure for a given energy range. The result is given in 1/m.
        """
        self.get_density()
        if density is None:
            density = float(self.density.subs(self.subs).n())

        if f1f2 is None:
            f1f2 = self.get_f1f2_isotropic(energy, table=table, fwhm_ev=fwhm_ev)

        index_refraction = complex(0)

        for label in f1f2:
            element = self.elements[label]
            Z = elements.Z[element]

            index_refraction += (Z + f1f2[label]) * self.multiplicity[label]


        index_refraction *= self.electron_radius/(2*np.pi) \
                         * (self.eV_A*1e-10/energy)**2 \
                         * density * 1e6 * self.avogadro / self.total_weight

        self.index_of_refraction = index_refraction

        const = 10135467.657934014 # 2*eV/c/hbar
        mu = index_refraction.imag * const * energy
        return mu





    def DAFS(self, energy, miller, DD=False, DQ=False, Temp=True, psi=0,
             func_output=False, fwhm_ev=None, table="Sasaki", channel="ss",
             simplify=True, subs=True, force_refresh=True, Uaniso=True):
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
        #if not hasattr(self, "F_0") or force_refresh:
        #    self.calc_structure_factor(miller, DD=DD, DQ=DQ, Temp=Temp,
        #                               subs=subs, evaluate=subs, Uaniso=Uaniso)

        miller = tuple(miller)#map(int, miller))
        assert len(miller)==3, "Input for `miller` index must be 3-tuple of int"

        oldmiller = self.hkl()

        self.hkl(*miller)

        self.f.clear()
        f = self.f

        #if not isinstance(energy, np.ndarray):
        energy = np.array(energy, dtype=float, ndmin=1)

        ### get dispersion correction:
        f1f2 = self.get_f1f2_isotropic(energy, fwhm_ev, table)



        ### get structure factor:
        if DD or DQ:
            if hasattr(self, "E") and miller==oldmiller:
                pass
            else:
                self.calc_structure_factor(miller, DD=DD, DQ=DQ, Temp=Temp,
                                       subs=subs, evaluate=subs, Uaniso=Uaniso)
                self.calc_scattered_amplitude(simplify=simplify, DD=DD, DQ=DQ,
                                              subs=subs)
            Feval = self.E[channel]
            f_aniso = dict()
            for sym in Feval.atoms(sp.Symbol):
                if not sym.name.startswith("f_"):
                    continue
                label = sym.name.split("_")[1]
                component = "_".join(sym.name.split("_")[2:])
                if component[:2] not in ["dd", "dq"]:
                    continue
                if not label in f_aniso and label in self.f_aniso_func:
                    func = self.f_aniso_func[label]
                    f_aniso[label] = named2darray(func(energy).T, func.components)
                if component in f_aniso[label]:
                    pass
                elif component[:2]=="dq":
                    ind = component.split("_")[1]
                    component = component[:3] + "%s%s%i"%("xyz"[int(ind[2])-1], ind[1], ["x","y","z"].index(ind[0])+1)
                f[sym] = f_aniso[label][component]

            f[self.energy] = energy
            f[self.S["psi"]] = psi

            #Feval = self.F_0.subs(self.subs.items()).n().expand() # self.F_0.subs(self.subs).n().expand()
            if simplify:
                Feval = Feval.simplify()
            if subs:
                Feval = Feval.subs(self.subs.items()).n().expand()
            self.Feval = Feval
            self.F_0_func = makefunc(Feval)
        else:
            if hasattr(self, "F_0") and \
               (self.F_0.has(*self.miller) or miller==oldmiller) and not \
               force_refresh:
                if self.DEBUG:
                    print("Using cached SF")
            else:
                F0 = self.calc_structure_factor(miller, DD=DD, DQ=DQ, Temp=Temp,
                                   subs=subs, evaluate=subs, Uaniso=Uaniso)
                #subit = self.subs.iteritems()
                #Feval = self.F_0.n().expand()


        q = float(self.qfunc.dictcall(self.subs).n())
        if self.DEBUG:
            print("Q-value =", q)
        if self.f0.get("__q__") != q:
            self.f0.clear()
            self.f0["__q__"] = q

        for label in self.AU_formfactors:
            ffsymbol = self.AU_formfactors[label]
            ion = self.get_ion(label)
            if not ion in self.f0:
                if self.DEBUG:
                    print("Calculating nonresonant scattering amplitude for %s"%ion)
                self.f0[ion] = self.f0func(ion, q)

            f[ffsymbol] = f1f2[label] + self.f0[ion]

        if Uaniso:
            for label in self.U:
                if label in self.Uaniso:
                    Uval = self.Uaniso[label]
                else:
                    Uval = sp.eye(3) * self.Uiso[label]
                if self.DEBUG:
                    print("Using U values for %s: %s"%(label, str(Uval)))
                vald = zip(sp.flatten(self.U[label]), sp.flatten(Uval))
                for (k,v) in vald:
                    if isinstance(k, sp.Symbol):
                        self.subs[k] = float(v)

        f.update(self.subs)



        if func_output:
            return self.F_0_func
        else:
            return self.F_0_func.dictcall(f)


    def get_equivalent_vectors(self, v):
        """
            Returns all reciprocal lattice vectors that are
            symmetrically equivalent to v
        """
        v = sp.Matrix(v)
        return set([tuple(G[:,:3].T * v) for G in self.generators])


    def iter_rec_space(self, qmax, independent=True):
        """

            Returns an iterator over all Bragg reflections of the structures
            as 3-tuples. The maximum value 2*sin(theta)/lambda is defined by
            qmax. If `independent` is True, only those reflections will be
            included that are symetrically inequivalent.

        """
        qmax = abs(qmax)
        hmax = int(self.subs[self.a] * qmax)
        kmax = int(self.subs[self.b] * qmax)
        lmax = int(self.subs[self.c] * qmax)
        hind = range(hmax, -hmax-1, -1)
        kind = range(kmax, -kmax-1, -1)
        lind = range(lmax, -lmax-1, -1)
        iter1 = itertools.product(hind, kind, lind)
        self._Rdone = set()
        kwargs = self.subs.copy()
        kwargs.update(zip(self.miller, self.miller))
        qfunc = makefunc(self.qfunc.dictcall(kwargs).n(), np.math)
        self._qfunc = qfunc
        def helper(R):
            if qfunc(*R) > qmax:
                return False
            if R in self._Rdone:
                return False
            if independent:
                self._Rdone.update(self.get_equivalent_vectors(R))
            return True
        return ifilter(helper, iter1)



    def calc_ADP_from_temperature(self, label, temperature,
                    debye_temperature=None, einstein_temperature=None,
                    mass=None, return_value="U"):
        """
            Sets the isotropic mean square displacement (unit_cell.U) of an
            atom according to the Einstein model and a given temperature.
            If the mass is not given, it is taken from the database.

            Inputs:
                label : str
                    the label of the atom
                temperature : float
                    the temperature in kelvin
                debye_temperature : float
                    the characteristic frequency in terms of the Debye
                    model
                einstein_temperature : float
                    the characteristic frequency in terms of the Einstein
                    model
                mass : float (optional)
                    the mass of the atom in atomic mass units
                component : 2-tuple
                    the index of the tensor component described.
                    If `None` the isotropic displacement is used
                return_value : str
                    Defines which value to return. Can be one of:
                        `U`     - The U parameter as used in .cif files
                        `B`     - The B parameter as used in .cif files
                        `msd`   - The isotropic mean square displacement
                    B = 8 * pi**2 * U
                    msd = U_isotropic * 3
            Returns:
                The isotropic mean square displacement in Angstrom^2
        """
        if not label in self.elements:
            raise ValueError("Atom labelled `%s` not found in unit_cell"%label)
        # http://dx.doi.org/10.1107/S0108767308031437
        element = self.elements[label]

        if mass != None:
            self.masses[element] = mass
        elif not element in self.masses:
            self.masses[element] = xi.get_element(element)[1]

        m = self.masses[element]

        if debye_temperature != None:
            model = "debye"
            self.debye_temperature[label] = debye_temperature
        elif einstein_temperature != None:
            model = "einstein"
            self.einstein_temperature[label] = einstein_temperature
        elif label in self.debye_temperature:
            model = "debye"
        elif label in self.einstein_temperature:
            model = "einstein"
        else:
            raise ValueError("No characteristic temperature given. "
              "Please specify `debye_temperature` or `einstein_temperature`")
        if self.DEBUG:
            print("Using %s model for ADP calculation of %s"%(model, label))

        Tc = getattr(self, "%s_temperature"%model)[label]

        if hasattr(Tc, "shape") and Tc.shape==(3,3):
            Tc = np.array(Tc).astype(float)
        else:
            Tc = float(Tc)

        if model == "einstein":
            omega = Tc * self.boltzmann / self.hbar
            U =   self.hbar/(2. * m * self.u * omega ) \
                   / np.tanh( self.hbar*omega / (2*self.boltzmann*temperature)) *1e10**2
        elif model == "debye":
            # 10.1007/978-3-642-74291-0 Eq. (3.2.24)
            # B = 8 * pi**2 * U
            U =  1e10**2 * 3 * self.hbar**2 / (m * self.u * Tc * self.boltzmann) * \
                  ( debye_phi(Tc/temperature) / (Tc/temperature) + 0.25)

        if hasattr(U, "shape") and U.shape==(3,3):
            self.Uaniso[label] = sp.Matrix(U)
        else:
            self.Uiso[label] = U
        if return_value=="msd":
            return U * 3
        elif return_value=="B":
            return U * 8 * np.pi**2
        else:
            return U

    def set_temperature(self, temperature):
        """
            Sets the isotropic mean square displacement (unit_cell.U) of ech
            atom according to the Einstein model and a given temperature.

            Before using it, the characteristic frequency in the einstein
            model needs to be given via the unit_cell.calc_ADP_from_temperature
            method.

            Inputs:
                temperature : float
                    the temperature in kelvin
        """
        for label in self.elements:
            self.calc_ADP_from_temperature(label, temperature)



    def simplify(self): # a longshot
        if not hasattr(self, "E"): return
        for pol in self.E:
            currlen = sp.count_ops(str(self.E[pol]))
            for i in range(3):
                print(i)
                for new in (sp.expand_mul(self.E[pol]), self.E[pol].expand(trig=True), sp.expand_complex(self.E[pol]).expand()):#, sp.trigsimp(self.E[pol])):
                    print(currlen)
                    if sp.count_ops(str(new))<currlen:
                        currlen=sp.count_ops(str(new))
                        self.E[pol] = new
        return self.E

    def eval_AAS(self, energy=None, table="Sasaki"):
        if energy!=None:
            self.subs[self.energy] = energy
        else:
            energy = float(self.subs[self.energy])
        self.d = 1./self.qfunc.dictcall(self.subs)
        q = self.qfunc.dictcall(self.subs).n()
        for label in self.AU_formfactors:
            ffsymbol = self.AU_formfactors[label]
            element = self.elements[label]
            ion = self.get_ion(label)
            Z = elements.Z[element]
            f0 = self.f0func(ion, q)
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
        for pol in self.E:
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
        if energy!=None:
            self.subs[self.energy] = energy
        else:
            energy = float(self.subs[self.energy])
        self.calc_structure_factor(miller, Temp=Temp)
        self.transform_structure_factor(AAS=False)

        done = []
        q = self.qfunc.dictcall(self.subs).n()
        for label in self.AU_formfactors:
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
            ion = self.get_ion(label)
            f0 = float(self.f0func(ion, q))

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


    def get_stereographic_projection(self, pole, azi_ref, energy, tthmin=0, 
                                     tthmax=sp.pi, Ithresh=0.01, verbose=False):
        """
            Calculates the 2D positions of all reciprocal lattice points on a
            stereographic projection in polar coordinates (r, phi).

            Arguments:
                pole : hkl of a reciprocal lattice point parallel to the pole
                azi_ref : hkl of the reciprocal lattice vector marking the
                          azimuthal zero position

            Returns: r, phi, intensity
                Nd arrays of the positions
        """
        latt_par = self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        lattice = dict([(k, self.subs[k]) for k in latt_par if k in self.subs])

        Gc_f = makefunc(self.Gc.subs(lattice), mathmodule="sympy")
        V = self.V.subs(lattice)

        pole_c = Gc_f(*pole).normalized().n()
        azi_ref_c = Gc_f(*azi_ref).normalized().n()
        azi_ref2_c = pole_c.cross(azi_ref_c).n().normalized()


        Lambda = 12398./energy
        qmax = float(2*sp.sin(tthmax/2)/Lambda)

        self.calc_structure_factor(Temp=False,
                                   Uaniso=False,
                                   DD=False,
                                   DQ=False,
                                   subs=True,
                                   evaluate=True)

        th_f = makefunc(self.theta)


        results = []
        for R in self.iter_rec_space(qmax, independent=False):
            if R == (0,0,0):
                continue
            if verbose:
                print(R, end=", ")

            tth = th_f(energy, *R)*2
            if tth < tthmin or tth > tthmax:
                continue

            F = self.DAFS(energy, R, force_refresh=False) / V
            I = abs(F)**2

            if I < Ithresh:
                continue
            Gc = Gc_f(*R)
            Gc_vert = Gc.dot(pole_c)
            Gc_ip = Gc - Gc_vert * pole_c
            Gc_x = float(Gc_ip.dot(azi_ref_c))
            Gc_y = float(Gc_ip.dot(azi_ref2_c))

            theta = sp.acos(Gc_vert / Gc.norm())
            r = float(sp.sin(theta))
            if Gc_x==Gc_y==0:
                phi = 0
            else:
                phi = sp.atan2(Gc_y, Gc_x)


            results.append((r, phi, I, tth, R[0], R[1], R[2]))

        if verbose:
            print("")

        dtype = [('r', 'f4'), ('phi', 'f4'), ('I', 'f4'), ('tth', 'f4'), ('h', 'i4'), ('k', 'i4'), ('l', 'i4')]
        results = np.array(results, dtype=dtype)

        return results


    def calc_reflection_angles(self, orientation, energy = None, **kwargs):
        """
            Calculates the angles of Bragg reflections relative to the surface
            of a crystal for given orientation (cut).
            Returns a (theta, psi)-tuple, where theta is the angle between the
            incident beam and the surface and psi is the azimuthal position.
        """
        if energy is None:
            if self.energy in self.subs:
                self.subs.pop(self.energy)
        else:
            self.subs[self.energy] = energy
        miller = list(sp.symbols("h,k,l", integer=True))
        for i in range(3):
            if miller[i].name in kwargs:
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
        if energy is None:
            if self.energy in self.subs:
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
            print(key)
            coordfunc[key]  = lambdify(("psi_2", "epsilon") + kwvar, [psi.subs(self.subs), theta.subs(self.subs)], "numpy")
        def coordinates(miller, energy, psi2):
            key = zip(("h", "k", "l"), miller)
            kwargs = dict(filter(lambda x:x[1]!=0, key))
            key = map(lambda x: x[0] if x[1]!= 0 else "0", key)
            key = "".join(key)
            return coordfunc[key](psi2, energy, **kwargs)

        return coordinates

    def ThreeBeamDiffraction(self,miller, psi, energy, qmax=1, verbose=True, **kwargs):
        """
            Calculates azimuthal (Renninger) scan profiles in the three
            beam approximation
            http://dx.doi.org/10.1107/S0108767396011117
        """
        SF_defaults = dict(DD=False, DQ=False, subs=True, evaluate=True)
        SF_defaults.update(kwargs)

        DAFS_kw = ("DD", "DQ", "Temp", "fwhm_ev", "table", "simplify", "subs", "Uaniso")
        DAFS_kw = dict([(k,v) for k in kwargs.items() if k in DAFS_kw])

        AAS = SF_defaults["DD"] or SF_defaults["DQ"]
        print("Calculating all structure factors...")
        self.calc_structure_factor(**SF_defaults)

        F_0 = self.DAFS(energy, (0,0,0), force_refresh=False, **DAFS_kw) # transmitted beam
        F_0h = self.DAFS(energy, miller, force_refresh=False, **DAFS_kw) # align for primary reflection
        print("Transforming to diffractometer system...")
        self.transform_structure_factor(AAS=AAS, subs=SF_defaults["subs"], simplify=False)

        psi = np.array(psi, dtype=float, ndmin=1)[:,None] # rotation around lattice plane normal
        psisym = sp.Symbol("psi", real=True) # the same as a symbol
        F_0_sym = sp.Symbol("F_0", complex=True) # a symbol for transmitted beam
        theta = self.theta_degrees(None, *miller, rad=True) # bragg angle

        sigma_0 = sp.Matrix([0,0,1]) # perpendicular polarization
        pi_0 = sp.Matrix([sp.cos(theta),  sp.sin(theta), 0]) # parallel polarization incident beam
        pi_h = sp.Matrix([sp.cos(theta), -sp.sin(theta), 0]) # parallel polarization scattered beam

        hc = sp.Symbol("hc", real=True, positive=True)
        k = self.energy / self.eV_A # wavevector E/hc

        vec_k_i = sp.Matrix([-sp.sin(theta), sp.cos(theta), 0]) # direction of incident wavevector
        vec_k_s = sp.Matrix([ sp.sin(theta), sp.cos(theta), 0]) # direction of scattered wavevector
        self.vec_k_i, self.vec_k_s = vec_k_i, vec_k_s
        # introduce rotational matrix
        self.Psi = Psi = np.array([[1,            0,           0],
                                   [0,  sp.cos(psisym), sp.sin(psisym)],
                                   [0, -sp.sin(psisym), sp.cos(psisym)]])

        subs = [(sym, self.subs[sym]) for sym in self._metrik if sym in self.subs]
        self.g_psi = g_psi = Psi * self.Q * self.Gc.subs(subs) # general 2nd reflection, symbolic

        print(os.linesep+"rec. lattice coords to diffractometer coords:")
        sp.pprint(self.Q)

        K_g = k*vec_k_i + g_psi # wave vector for reflection g (K(h_m)), symbolic
        Vc = self.V.subs(self.subs)
        Gamma = self.electron_radius*1e10 / (k**2 * sp.pi * Vc)
        R_g = k**2 / K_g.dot(K_g) - (1 + Gamma*F_0_sym) # symbolic, -Gamma*F(0) = chi(0)

        if self.DEBUG:
            self.R_g = R_g
            self.Gamma = Gamma

        self.K_g = K_g
        pi_g = K_g.cross(sigma_0)
        sigma_g = pi_g.cross(K_g)

        self.pi_g = pi_g = pi_g / sp.sqrt(pi_g[0]**2 + pi_g[1]**2 + pi_g[2]**2) # real norm
        self.sigma_g = sigma_g = sigma_g / sp.sqrt(sigma_g[0]**2 + sigma_g[1]**2 + sigma_g[2]**2) # real norm

        if self.DEBUG:
            print(map(sp.count_ops, sigma_g), map(sp.count_ops, pi_g))
        if kwargs["simplify"]:
            print("Trying to simplify polarization vector expressions...")
            pi_g.simplify()
            sigma_g.simplify()
        if self.DEBUG:
            print(map(sp.count_ops, sigma_g), map(sp.count_ops, pi_g))

        # polarization matrizes:
        #alpha_0h_ss = 1
        alpha_0h_pp = pi_0.dot(pi_h)

        alpha_0g = sp.Matrix([[sigma_0.dot(sigma_g),              0],
                              [   pi_0.dot(sigma_g), pi_0.dot(pi_g)]])

        alpha_gh = sp.Matrix([[sigma_g.dot(sigma_0), sigma_g.dot(pi_h)],
                              [   pi_g.dot(sigma_0),    pi_g.dot(pi_h)]])

        alpha_0h2 = alpha_0g*alpha_gh

        #ecmplx = sp.Symbol("epsilon_c", complex=True)
        #R_g_f = makefunc(R_g.subs(self.energy, ecmplx), "numpy")
        R_g_f = makefunc(R_g, "numpy")

        vectorizer = makeufunc
        alpha_0h_pp_f = vectorizer(alpha_0h_pp)
        #alpha_gh_f = makefunc(alpha_gh, "numpy")
        alpha_gh_fs = vectorizer((alpha_gh[0,0]**2 + alpha_gh[1,0]**2))
        alpha_gh_fp = vectorizer((alpha_gh[0,1]**2 + alpha_gh[1,1]**2))
        alpha_0h2_fss = vectorizer((alpha_0h2)[0,0])
        alpha_0h2_fsp = vectorizer((alpha_0h2)[0,1])
        alpha_0h2_fps = vectorizer((alpha_0h2)[1,0])
        alpha_0h2_fpp = vectorizer((alpha_0h2)[1,1])

#        if simplify:
#            print("Trying to simplify alpha matrices...")
#            alpha_gh_fs   = alpha_gh_fs.simplify()
#            alpha_gh_fp   = alpha_gh_fp.simplify()
#            alpha_0h2_fss = alpha_0h2_fss.simplify()
#            alpha_0h2_fsp = alpha_0h2_fsp.simplify() # too big
#            alpha_0h2_fps = alpha_0h2_fps.simplify()
#            alpha_0h2_fpp = alpha_0h2_fpp.simplify()


        if self.DEBUG:
            self.alpha_gh_fs = alpha_gh_fs
            self.alpha_gh_fp = alpha_gh_fp
            self.alpha_0h2_fss = alpha_0h2_fss
            self.alpha_0h2_fsp = alpha_0h2_fsp
            self.alpha_0h2_fps = alpha_0h2_fps
            self.alpha_0h2_fpp = alpha_0h2_fpp

        Gamma = makefunc(Gamma, "numpy")(epsilon=energy)

        if verbose:
            print("chi0 = ",(-Gamma * F_0))


        #energyc = np.array(energy, dtype=complex)
        subs = dict(epsilon=energy,
                    #epsilon_c=energyc,
                    F_0=F_0, psi=psi)

        for i in range(3):
            subs[self.miller[i]] = miller[i]
        #self._subs = subs

        Ns = Np = 1.
        F_eff = np.array([0,0,0,1])[:,None,None] * alpha_0h_pp_f(energy) # setup array for pp
        F_eff[0] += 1 #ss channel
        F_eff = F_eff * np.ones_like(psi) * F_0h
        #print F_eff.shape, F_eff.dtype
        #self.R_h = R_h = 1./R_g_f.dictcall(subs)
        R_h = 1./(- Gamma*F_0.imag) # lets assume we are in the max. for now. Add detuning later
        _Fcache = dict()
        _Fmaxcache = dict()
        for g in self.iter_rec_space(qmax, independent=False):
            if g==(0,0,0):
                continue
            if verbose:
                print(g, end="")
            subs.update(zip(self.miller, g))
            hg = tuple(miller[i] - g[i] for i in range(3))
            if hg==(0,0,0):
                continue
            #F_g  = self.DAFS(energy,  g, force_refresh=False)
            if g not in _Fcache:
                self.f.update(zip(self.miller, g))
                F_g  = self.F_0_func.dictcall(self.f)
                for _R in self.get_equivalent_vectors(g):
                    _Fcache[_R] = F_g
                    _Fmaxcache[_R] = abs(F_g).max()
            if hg not in _Fcache:
                self.f.update(zip(self.miller, hg))
                F_hg  = self.F_0_func.dictcall(self.f)
                for _R in self.get_equivalent_vectors(hg):
                    _Fcache[_R] = F_hg
                    _Fmaxcache[_R] = abs(F_hg).max()

            if _Fmaxcache[g]<1e-6 or _Fmaxcache[hg]<1e-6:
                if verbose:
                    print("skipped")
                continue
            elif verbose:
                print("")

            F_g  = _Fcache[g]
            F_hg  = _Fcache[hg]


            if self.DEBUG:
                smth = np.array([
                                 alpha_0h2_fss.dictcall(subs),
                                 alpha_0h2_fsp.dictcall(subs),
                                 alpha_0h2_fps.dictcall(subs),
                                 alpha_0h2_fpp.dictcall(subs)
                                 ])
                #print("%i NaN values in F12"%np.isnan(F12).sum())
                for i in range(4):
                    print("%i NaN values in alpha_%i"%(np.isnan(smth[i]).sum(), i))
                self._subs = subs

            R_g = 1./R_g_f.dictcall(subs)
            #print R_g.max(), R_g.min()
            F12 = Gamma * R_g * F_g * F_hg

            F_eff[0] += alpha_0h2_fss.dictcall(subs) * F12
            F_eff[1] += alpha_0h2_fsp.dictcall(subs) * F12
            F_eff[2] += alpha_0h2_fps.dictcall(subs) * F12
            F_eff[3] += alpha_0h2_fpp.dictcall(subs) * F12

            #print alpha_0h2_fss.dictcall(subs).shape,alpha_0h2_fss.dictcall(subs).dtype,F12.shape,F_eff.shape,F_eff.dtype


            fac = (Gamma*F_hg)**2 * R_g * R_h
            Ns -= alpha_gh_fs.dictcall(subs) * fac
            Np -= alpha_gh_fp.dictcall(subs) * fac

        self.F_eff = F_eff
        self.Ns = Ns
        self.Np = Np
        F_eff *= Gamma * R_h
        F_eff.resize((2,2) + F_eff.shape[1:])
        F_eff[:,0] /= Ns
        F_eff[:,1] /= Np

        return F_eff






    def ThreeBeamDiffraction_slow(self,miller, psi, energy, qmax=1, verbose=True, **SF_kwargs):
        """
            Calculates azimuthal (Renninger) scan profiles in the three
            beam approximation
            http://dx.doi.org/10.1107/S0108767396011117
        """
        SF_defaults = dict(DD=False, DQ=False, subs=True, evaluate=True)
        SF_defaults.update(SF_kwargs)
        AAS = SF_defaults["DD"] or SF_defaults["DQ"]
        print("Calculating all structure factors...")
        self.calc_structure_factor(**SF_defaults)

        F_0 = self.DAFS(energy, (0,0,0), force_refresh=False) # transmitted beam
        F_eff = self.DAFS(energy, miller, force_refresh=False) # align for primary reflection
        print("Transforming to diffractometer system...")
        self.transform_structure_factor(AAS=AAS, subs=SF_defaults["subs"], simplify=False)

        psi = np.array(psi, dtype=float, ndmin=1)[:,None] # rotation around lattice plane normal
        psisym = sp.Symbol("psi", real=True) # the same as a symbol
        F_0_sym = sp.Symbol("F_0", complex=True, real=False, imaginary=False) # a symbol for transmitted beam
        theta = self.theta_degrees(None, *miller, rad=True) # bragg angle

        sigma_0 = sp.Matrix([0,0,1]) # perpendicular polarization
        pi_0 = sp.Matrix([sp.cos(theta),  sp.sin(theta), 0]) # parallel polarization incident beam
        pi_h = sp.Matrix([sp.cos(theta), -sp.sin(theta), 0]) # parallel polarization scattered beam

        k = self.energy / self.eV_A # wavevector

        vec_k_i = sp.Matrix([-sp.sin(theta), sp.cos(theta), 0]) # direction of incident wavevector
        vec_k_s = sp.Matrix([ sp.sin(theta), sp.cos(theta), 0]) # direction of scattered wavevector
        self.vec_k_i, self.vec_k_s = vec_k_i, vec_k_s
        # introduce rotational matrix
        self.Psi = Psi = np.array([[1,            0,           0],
                                   [0,  sp.cos(psisym), sp.sin(psisym)],
                                   [0, -sp.sin(psisym), sp.cos(psisym)]])

        subs = [(sym, self.subs[sym]) for sym in self._metrik if sym in self.subs]
        self.g_psi = g_psi = Psi * self.Q * self.Gc.subs(subs) # general 2nd reflection, symbolic

        K_g = k*vec_k_i + g_psi # wave vector for reflection g (K(h_m)), symbolic
        Vc = self.V.subs(self.subs)
        Gamma = self.electron_radius*1e10 / (k**2 * sp.pi * Vc)
        R_g = k**2 / K_g.dot(K_g) - (1 + Gamma * F_0_sym) # symbolic
        self.R_g = R_g



        #K_g_norm = K_g.normalized()

        #self.K_g_s = K_g.dot(sigma_0)
        #self.K_g_p0 = K_g.dot(pi_0)
        #self.K_g_ph = K_g.dot(pi_h)

        self.pi_g = pi_g = K_g.cross(sigma_0).normalized()
        self.sigma_g = sigma_g = pi_g.cross(K_g).normalized()

        One = sp.Symbol("One") # dummy
        Ones = np.ones_like(energy)
        # polarization matrizes:
        alpha_0h = sp.Matrix([[One,          0],
                              [0, pi_0.dot(pi_h)]])

        alpha_0g = sp.Matrix([[sigma_0.dot(sigma_g),              0],
                              [   pi_0.dot(sigma_g), pi_0.dot(pi_g)]])

        alpha_gh = sp.Matrix([[sigma_g.dot(sigma_0), sigma_g.dot(pi_h)],
                              [   pi_g.dot(sigma_0),    pi_g.dot(pi_h)]])

        unity = sp.Matrix([[One,   0],
                           [  0, One]])


        R_g_f = makefunc(R_g, "numpy")
        alpha_0h_f = makefunc(alpha_0h, "numpy")
        alpha_0g_f = makefunc(alpha_0g, "numpy")
        alpha_gh_f = makefunc(alpha_gh, "numpy")
        #alpha_0h2_f = makefunc(alpha_0g*alpha_gh, "numpy")
        Gamma = makefunc(Gamma, "numpy")(epsilon=energy)
        unity = makefunc(unity, "numpy")

        print("chi0 = ",(-Gamma * F_0))
        self.alpha_0h = alpha_0h
        self.alpha_0g = alpha_0g
        self.alpha_gh = alpha_gh

        energy = np.array(energy, dtype=complex)
        subs = dict(One=Ones, epsilon=energy,
                    h=miller[0], k=miller[1], l=miller[2],
                    F_0=F_0, psi=psi)

        Ns = Np = 1.
        F_eff = alpha_0h_f.dictcall(subs) * unity(F_eff)
        R_h = 1./R_g_f.dictcall(subs)
        for g in self.iter_rec_space(qmax, independent=False):
            if g==(0,0,0):
                continue
            if verbose:
                print(g, end="")
            subs.update(zip(self.miller, g))
            hg = tuple(miller[i] - g[i] for i in range(3))
            if hg==(0,0,0):
                continue
            #F_g  = self.DAFS(energy,  g, force_refresh=False)
            #F_hg = self.DAFS(energy, hg, force_refresh=False)
            self.f.update(zip(self.miller, g))
            F_g  = self.F_0_func.dictcall(self.f)
            self.f.update(zip(self.miller, hg))
            F_hg = self.F_0_func.dictcall(self.f)
            if abs(F_g).max()<1e-6 or abs(F_hg).max()<1e-6:
                if verbose:
                    print("skipped")
                continue
            else:
                if verbose:
                    print("")
            R_g = 1./R_g_f.dictcall(subs)
            #F_eff += alpha_0h2_f.dictcall(subs)* unity(Gamma * R_g*F_g*F_hg)
            F_eff += alpha_0g_f.dictcall(subs)*alpha_gh_f.dictcall(subs) \
                   * unity(Gamma * R_g*F_g*F_hg)

            #print alpha_0g_f.dictcall(subs), alpha_gh_f.dictcall(subs), unity(Gamma * R_g*F_g*F_hg)

            fac = (Gamma*F_hg)**2 * R_g * R_h
            tmp = alpha_gh_f.dictcall(subs)
            Ns -= (tmp[:,0].A**2).sum() * fac
            Np -= (tmp[:,1].A**2).sum() * fac

        self.F_eff = F_eff
        self.Ns = Ns
        self.Np = Np
        F_eff = F_eff * unity(Gamma * R_h)
        F_eff[0,0] /= Ns
        F_eff[1,0] /= Ns
        F_eff[0,1] /= Np
        F_eff[1,1] /= Np


        return F_eff


    def XRD_pattern(self, x, energy, xaxis="theta", qmax=None, width=0.01, grainsize=44.,
                           Lfac = True, Pfac = True, poling=None, verbose=False):
        """
            Calculates the diffraction intensity for all reflections in
            a certain range for theta - the Bragg angle.

            Inputs:
                x : numpy.ndarray((n,), float)
                    Array containing the values of Bragg angles. See
                    `xaxis` argument
                energy : float
                    Photon energy in electron volt
                xaxis : str
                    Defines how to process the `x` input
                    One of:
                        theta  (in degrees)
                        qcryst (inverse AA)
                qmax : float
                    maximum value of 2*sin(theta)/lambda taken into
                    account
                width : float
                    FWHM of experimental broadening in degrees
                grainsize : float
                    average grain size of the powder in microns for
                    additional broadening according to Scherrer equation
                poling : 3d array
                    vector indication preferred orientation

            Outputs:
                Intensity : numpy.ndarray((n,), float)
                    Array containing the Intensity value for
                    each theta value in arbitrary units
        """
        def LP(theta, degrees=True):
            if degrees:
                theta = np.radians(theta)
            if not Lfac:
                L = 1
            elif Lfac == "psd":
                L = 1. / np.sin(2*theta)
            else:
                L = 1./(4.*np.sin(theta)**2*np.cos(theta))
            P = (1 + np.cos(2*theta)**2)/2. if Pfac else 1
            return L*P
        self._LP = LP

        energy = np.array(energy, ndmin=1)
        Lambda = self.eV_A / energy

        if xaxis == "theta":
            th_arr = np.radians(x)
        elif xaxis == "qcryst":
            th_arr = np.arcsin(x * Lambda / 2)
        else:
            raise ValueError("Input for `xaxis` not understood.")

        width = np.radians(width)**2
        grainsize *= 1e4 # microns to angstrom
        if qmax == None:
            qmax = (2 / Lambda * np.sin(th_arr.max()))
            if verbose:
                print("Using max. value 2sin(th)/lambda: %f"%qmax)
        Int = np.zeros_like(th_arr+energy)
        V = float(self.V.subs(self.subs))
        self.calc_structure_factor(subs=True, evaluate=True)
        thetafunc = makefunc(self.theta)
        for R in self.iter_rec_space(qmax):
            #if R == (0,0,0):
            #    continue
            if verbose:
                print(R)
            theta = thetafunc(energy, *R)
            if np.all(np.isnan(theta)):
                continue
            theta = np.float_(theta)
            #print self.qfunc.dictcall(self.subs)
            F = self.DAFS(energy.ravel(), R, force_refresh=False) / V
            F = F.reshape(energy.shape)
            M = len(self.get_equivalent_vectors(R))
            I = np.float_(abs(F)**2 * M) * LP(th_arr, False)
            if poling is not None:
                v = np.array(list(self.get_equivalent_vectors(R)), dtype=float).T
                fac = np.dot(poling, v)/np.linalg.norm(v, axis=0)/np.linalg.norm(poling)
                I *= (1+fac).mean()/2.
            width2 = (0.9 * 12398.4 / ( 2 * energy * grainsize * np.cos(theta)))**2
            w = width + width2 # broadening
            w = np.sqrt(w) # /2.3548
            #print R, w, theta
            I = pvoigt(th_arr, theta, I, w, eta=(width2/(width2 + width)))/w
            I[np.isnan(I)] = 0
            Int += I
        return Int
