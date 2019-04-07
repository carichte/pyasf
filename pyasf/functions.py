import os
import sys
import itertools
import string
import sympy as sp
import numpy as np
import types

try:
    import mpmath
    sp.mpmath = mpmath
except:
    pass

from sympy.utilities.autowrap import ufuncify
from sympy.utilities import lambdify

if sys.version_info[0]<3:
    import cPickle as pickle
    from urllib import urlopen, urlencode
else:
    import pickle
    from urllib.request import urlopen
    from urllib.parse import urlencode


SETTPATH = os.path.join(os.path.dirname(__file__), "settings.txt.gz")

decode = np.vectorize(bytes.decode)

def get_ITA_settings(sgnum):
    if os.path.isfile(SETTPATH):
        #settings = np.genfromtxt(SETTPATH, dtype="i2,O,O")
        settings = np.genfromtxt(SETTPATH, dtype="i2,O,O", delimiter=";", autostrip=True)
        ind = settings["f0"]==sgnum
        #print(settings["f1"][ind])
        return dict(zip(decode(settings["f1"][ind]), 
                        decode(settings["f2"][ind])
                      ))

    else:
        from lxml import etree
        print("Fetching ITA settings for space group %i"%sgnum)
        baseurl = "http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-getgen"
        params = urlencode({'gnum': sgnum, 'what':'gp', 'settings':'ITA Settings'}).encode()
        result = urlopen(baseurl, params)
        result = result.read()
        parser = etree.HTMLParser()
        tree = etree.fromstring(result, parser, base_url = baseurl)
        table = tree[1][3][6][0]
        settings = dict()
        for tr in table:
            children = tr[1].getchildren()
            if not children:
                continue
            elif children[0].tag=="a":
                a = children[0]
                trmat = tr[2][0].text
                setting = ""
                for i in a:
                    setting += i.text
                    if i.tail != None:
                        setting += i.tail
                if "origin" in setting:
                    setting = setting.split("[origin")
                    #print setting
                    setting = setting[0] + ":" + setting[1].strip("]")
                    
                setting = setting.replace(" ", "")
                #url = a.attrib["href"]
                #url = urllib.quote(url, safe="%/:=&?~#+!$,;'@()*[]")
                #settings[setting] = urllib.basejoin(baseurl, url)
                settings[setting.decode()] = trmat.decode()
    return settings


def fetch_ITA_generators(sgnum, trmat=None):
    """
        Retrieves all Generators of a given Space Group (sgnum) and for an
        unconventional setting `trmat' if given from
        http://www.cryst.ehu.es.
    """
    #url = "http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-getgen"
    url =  "http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-trgen"
    params = {'gnum': sgnum, 'what':'gp'}
    if trmat!=None:
        params["trmat"] = trmat
    params = urlencode(params).encode()
    result = urlopen(url, params)
    result = result.read()
    from lxml import etree
    parser = etree.HTMLParser()
    tree = etree.fromstring(result, parser, base_url = url)
    
    table = tree[1].find("center").findall("table")[1]
    
    generators=[]
    genlist = list(table)
    if table.find("tbody") != None:
        for tbody in table.findall("tbody"):
            genlist.extend(list(tbody))
    for tr in genlist:
        if not isinstance(tr[0].text, str) or not tr[0].text.isdigit():
            continue
        pre = tr[4][0][0][1][0].text
        generator = list(map(sp.S, pre.split()))
        generators.append(sp.Matrix(generator).reshape(3,4))
    return generators


def get_generators(sgnum=0, sgsym=None):
    """
        Retrieves all Generators of a given Space Group Number (sgnum) a local
        database OR from http://www.cryst.ehu.es and stores them into
        the local database.
        
        Inputs:
            sgnum : int
                Number of space groupt according to International Tables
                of Crystallography (ITC) A
            
            sgsym : str
                Full Hermann-Mauguin symbol according to ITC A.
                In some cases, more than one setting can be chosen for the 
                space group. Then it is possible to specify the desired 
                setting by giving the full Hermann-Mauguin symbol here.
                Otherwise the standard setting will be picked.
    """
    if isinstance(sgnum, str):
        if sgnum.isdigit():
            sgnum = int(sgnum)
        else:
            if sgsym==None:
                sgsym = str(sgnum)
            sgnum = 0
    if sgnum==0 and sgsym!=None and os.path.isfile(SETTPATH):
        #settings = np.genfromtxt(SETTPATH, dtype="i2,O,O")
        settings = np.genfromtxt(SETTPATH, dtype="i2,O,O", delimiter=";", autostrip=True)
        settings["f1"] = decode(settings["f1"])
        settings["f2"] = decode(settings["f2"])
        ind = settings["f1"] == sgsym
        if not ind.sum()==1:
            raise ValueError("Space group not found: %s"%sgsym)
        sgnum, _, trmat = settings[ind].item()
        #sgnum = settings["f0"][ind].item()
        #trmat = settings["f2"]][ind]
    elif isinstance(sgnum, int):
        if sgnum < 1 or sgnum > 230:
            raise ValueError("Space group number must be in range of 1...230")
        settings = get_ITA_settings(sgnum)
        #print(settings)
        setnames = " ".join(settings.keys())
        if len(settings)==1:
            sgsym = list(settings.keys())[0]
        elif sgsym!=None:
            if sgsym not in settings:
                raise ValueError(
                 "Invalid space group symbol (sgsym) entered: %s%s"\
                 "  Valid space group settings: %s"\
                 %(sgsym, os.linesep,setnames))
        else:
            print("Warning: Space Group #%i is ambigous:%s"\
                  "  Possible settings: %s "%(sgnum, os.linesep, setnames))
            settings_inv = dict([(v,k) for (k,v) in settings.items()])
            sgsym = settings_inv["a,b,c"]
            print("  Using standard setting: %s"%sgsym)
        trmat = settings[sgsym]
    else:
        raise ValueError("Integer required for space group number (`sgnum')")


    if SETTPATH.endswith(".gz"):
        from gzip import open as gzopen
        read = lambda fname: gzopen(fname, "rt")
    else:
        read = lambda fname: open(fname, "r")


    generators = []
    with read(SETTPATH) as fh:
        while True:
            line = fh.readline().split(";")
            if len(line)<2:
                continue
            if sgsym in line[1]:
                break
        while True:
            line = fh.readline()
            if not line.startswith("#"):
                break
            generators.append(sp.S(line.strip("# ").split(";")[1]))

    # fallback
    #generators = fetch_ITA_generators(sgnum, trmat)
    return generators
    

def gcd(*args):
    if len(args) == 1:
        return args[0]
    
    L = list(args)
    
    while len(L) > 1:
        a = L[len(L) - 2]
        b = L[len(L) - 1]
        L = L[:len(L) - 2]
        
        while a:
            a, b = b%a, a
        
        L.append(b)
    return abs(b)


def stay_in_UC(coordinate):
    if coordinate.has(sp.Symbol): return coordinate
    else: return coordinate%1
    

@sp.vectorize(0)
def hassymb(x):
    return x.has(sp.Symbol)


dictcall = lambda self, d: self.__call__(*[d.get(k, d.get(k.name, k)) for k in self.kw])

def makefunc(expr, mathmodule = "numpy", dummify=False, **kwargs):
    symbols = list(expr.atoms(sp.Symbol))
    symbols.sort(key=str)
    func = lambdify(symbols, expr, mathmodule, dummify=dummify, **kwargs)
    func.kw = symbols
    func.expr = expr
    func.kwstr = map(lambda x: x.name, symbols)
    func.dictcall = types.MethodType(dictcall, func)
    func.__doc__ = str(expr)
    return func

class makeufunc(object):
    def __init__(self, expr):
        symbols = list(expr.atoms(sp.Symbol))
        symbols.sort(key=lambda s: s.name)
        self.func = ufuncify(symbols, expr)
        self.kw = symbols
        self.expr = expr
        self.kwstr = map(lambda x: x.name, symbols)
        self.__doc__ = "f = f(%s)"%(", ".join(self.kwstr))
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    def dictcall(self, d):
        return self.func(*[d.get(k, d.get(k.name, k)) for k in self.kw])



def full_transform_old(Matrix, Tensor):
    """
        Transforms the Tensor to Representation in new Basis with given Transformation Matrix.
    """
    import numpy
    for i in range(Tensor.ndim):
        Axes = range(Tensor.ndim)
        Axes[0] = i
        Axes[i] = 0
        Tensor = numpy.tensordot(Matrix, Tensor.transpose(Axes), axes=1).transpose(Axes)
    return Tensor

def full_transform2(Matrix, Tensor):
    """
        Transforms the Tensor to Representation in new Basis with given Transformation Matrix.
    """
    for i in range(Tensor.ndim):
        Tensor = np.tensordot(Tensor, Matrix, axes=(0,0))
    return Tensor

def full_transform(Matrix, Tensor):
    """
        Transforms the Tensor to Representation in new Basis with given Transformation Matrix
        (but using somehow the transformed matrix :/).
    """
    Matrix = np.array(Matrix)
    Tensor = np.array(Tensor)
    dtype = np.find_common_type([],[Matrix.dtype, Tensor.dtype])
    Tnew = np.zeros_like(Tensor, dtype = dtype)
    for ind in itertools.product(*map(range, Tensor.shape)): # i
        for inds in itertools.product(*map(range, Tensor.shape)): #j
            Tnew[ind] += Tensor[inds] * Matrix[inds, ind].prod() # 
            #print Matrix[inds, ind], Tnew
    return Tnew


def get_cell_parameters(sg, sgsym = None):
        """
            Returns the general cell parameters for a lattice of given space group number sg.
        """
        import sympy as sp
        a, b, c, alpha, beta, gamma = sp.symbols("a, b, c, alpha, beta, gamma", real=True, positive=True, finite=True)
        
        if sg in range(1,3): 
            system = "Triclinic"
        elif sg in range(3,16):
            if isinstance(sgsym, str):
                trmat = get_ITA_settings(sg)[sgsym].split(",")
                rightangles = [i for i in range(3) if "b" not in trmat[i]]
            else:
                rightangles = [0,2]
            if 0 in rightangles: alpha=sp.S("pi/2")
            else: unique = "a"
            if 1 in rightangles: beta =sp.S("pi/2")
            else: unique = "b"
            if 2 in rightangles: gamma=sp.S("pi/2")
            else: unique = "c"
            system = "Monoclinic (unique axis %s)"%unique
        elif sg in range(16,75): 
            alpha=sp.S("pi/2")
            beta=sp.S("pi/2")
            gamma=sp.S("pi/2")
            system = "Orthorhombic"
        elif sg in range(75,143):
            b=a
            alpha=sp.S("pi/2")
            beta=sp.S("pi/2")
            gamma=sp.S("pi/2")
            system = "Tetragonal"
        elif sg in range(143,168):
            b=a
            if isinstance(sgsym, str) and sgsym.endswith(":r"):
                c = a
                beta = alpha
                gamma = alpha
                system = "Trigonal (rhombohedral setting)"
            else:
                alpha=sp.S("pi/2")
                beta=sp.S("pi/2")
                gamma=sp.S("2/3*pi")
                system = "Trigonal (hexagonal setting)"
        elif sg in range(168,195):
            b=a
            alpha=sp.S("pi/2")
            beta=sp.S("pi/2")
            gamma=sp.S("2/3*pi")
            system = "Hexagonal"
        elif sg in range(195,231):
            b=a
            c=a
            alpha=sp.S("pi/2")
            beta=sp.S("pi/2")
            gamma=sp.S("pi/2")
            system = "Cubic"
        else: 
            raise ValueError(
                "Invalid Space Group Number. Has to be in range(1,231).")
        print(system)
        return a, b, c, alpha, beta, gamma, system

def get_rec_cell_parameters(a, b, c, alpha, beta, gamma):
    import sympy as sp
    """
        Returns cell vectors in direkt and reciprocal space and reciprocal
        lattice parameters of given cell parameters (vectors in direct crystal
        fixed carthesian system).
    """
    
    # Metric Tensor:
    G = sp.Matrix([[a**2,               a*b*sp.cos(gamma), a*c*sp.cos(beta)],
                   [a*b*sp.cos(gamma), b**2,              b*c*sp.cos(alpha)],
                   [a*c*sp.cos(beta),  b*c*sp.cos(alpha), c**2]])
    
    G_r = G.inv() # reciprocal Metric
    G_r.simplify()
    
    # volume of crystall system cell in carthesian system
    # V = sp.sqrt(G.det())
    
    
    # reciprocal cell lengths
    ar = sp.sqrt(G_r[0,0])
    br = sp.sqrt(G_r[1,1])
    cr = sp.sqrt(G_r[2,2])
    
    #alphar = sp.acos(G_r[1,2]/(br*cr)).simplify()
    alphar = sp.acos((-sp.cos(alpha) + sp.cos(beta)*sp.cos(gamma))/(sp.Abs(sp.sin(beta))*sp.Abs(sp.sin(gamma))))
    betar  = sp.acos(G_r[0,2]/(ar*cr))
    #gammar = sp.acos(G_r[0,1]/(ar*br)).simplify()
    gammar = sp.acos((sp.cos(alpha)*sp.cos(beta) - sp.cos(gamma))/(sp.Abs(sp.sin(alpha))*sp.Abs(sp.sin(beta))))
    
    # x parallel to a* and z parallel to a* x b*   (ITC Vol B Ch. 3.3.1.1.1)
    #B =   sp.Matrix([[ar, br*sp.cos(gammar),  cr*sp.cos(betar)],
    #                 [0,  br*sp.sin(gammar), -cr*sp.sin(betar)*sp.cos(alpha)],
    #                 [0,  0,                  1/c]])
    #B_0 = sp.Matrix([[1,  sp.cos(gammar),  sp.cos(betar)],
    #                 [0,  sp.sin(gammar), -sp.sin(betar)*sp.cos(alpha)],
    #                 [0,  0,               1]])
    #V_0 = sp.sqrt(1 - sp.cos(alphar)**2 - sp.cos(betar)**2 - sp.cos(gammar)**2 + 2*sp.cos(alphar)*sp.cos(betar)*sp.cos(gammar))
    
    # x parallel to a and z parallel to a x b   (ITC Vol B Ch. 3.3.1.1.1)
    V0 = sp.sin(alpha) * sp.sin(beta) * sp.sin(gammar)
    M = sp.Matrix([[a, b*sp.cos(gamma),  c*sp.cos(beta)],
                   [0, b*sp.sin(gamma),  c*(sp.cos(alpha) - sp.cos(beta)*sp.cos(gamma))/sp.sin(gamma)],
                   [0,               0,  c*V0/sp.sin(gamma)]])
    Mi = sp.Matrix([[1/a,-1/(a*sp.tan(gamma)),  (sp.cos(alpha)*sp.cos(gamma) - sp.cos(beta))/(a*V0*sp.sin(gamma))],
                    [0,   1/(b*sp.sin(gamma)),  (sp.cos(beta)*sp.cos(gamma) - sp.cos(alpha))/(b*V0*sp.sin(gamma))],
                    [0,   0,                     sp.sin(gamma)/(c*V0)]])
    
    return (ar, br, cr, alphar, betar, gammar, M, Mi, G, G_r)

@np.vectorize
def debye_phi(v):
    if v==0:
        return 1.
    elif v < 1e-3:
        phi = v
    elif v > 1e2:
        phi = np.pi**2/6
    else:
        v = complex(v)
        phi = (sp.mpmath.fp.polylog(2, np.exp(v.real)) - v**2/2. + v*np.log(1-np.exp(v)) -  np.pi**2/6.)
    return (phi/v).real
#debye_phi_v = np.vectorize(debye_phi)

def pvoigt(x, x0, amp, fwhm, y0=0, eta=0.5):
    fwhm /= 2.
    return y0 + amp *    (eta  / (1+((x-x0)/fwhm)**2) 
                     + (1-eta) * np.exp(-np.log(2)*((x-x0)/fwhm)**2))



def axangle2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`
    taken and adapted from transforms3d package

    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.

    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation

    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = axis
    if not is_normalized:
        n = sp.sqrt(x*x + y*y + z*z)
        x = x/n
        y = y/n
        z = z/n
    c = sp.cos(angle); s = sp.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return sp.Matrix([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])



def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1 = sp.Matrix(v1).normalized()
    v2 = sp.Matrix(v2).normalized()
    return sp.acos(v1.dot(v2))



def rotation_from_vectors(v1, v2):
    """
        Find the rotation Matrix R that fullfils:
            R*v2 = v1

        Jur van den Berg,
        Calculate Rotation Matrix to align Vector A to Vector B in 3d?,
        URL (version: 2016-09-01): https://math.stackexchange.com/q/476311
    """
    v1 = sp.Matrix(v1).normalized()
    v2 = sp.Matrix(v2).normalized()

    ax = v1.cross(v2)
    s = ax.norm()
    c = v1.dot(v2)

    if c==1:
        return sp.eye(3)
    if c==-1:
        return -sp.eye(3)


    u1, u2, u3 = ax
    u_ = sp.Matrix(((  0, -u3,  u2), 
                    ( u3,   0, -u1),
                    (-u2,  u1,   0)))

    R = sp.eye(3) - u_ + u_**2 * (1-c)/s**2

    return R


