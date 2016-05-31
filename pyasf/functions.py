import os
import sympy as sp
from sympy.utilities.autowrap import ufuncify
from sympy.utilities import lambdify
import types
import urllib
import cPickle as pickle
import numpy as np
import itertools
import string

DBPATH = os.path.join(os.path.dirname(__file__), "space-groups.sqlite")
SETTPATH = os.path.join(os.path.dirname(__file__), "settings.txt")


def get_ITA_settings(sgnum):
    if os.path.isfile(SETTPATH):
        settings = np.genfromtxt(SETTPATH, dtype="i2,O,O")
        ind = settings["f0"]==sgnum
        return dict(zip(settings["f1"][ind], settings["f2"][ind]))
    else:
        from lxml import etree
        print("Fetching ITA settings for space group %i"%sgnum)
        baseurl = "http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-getgen"
        params = urllib.urlencode({'gnum': sgnum, 'what':'gp', 'settings':'ITA Settings'})
        result = urllib.urlopen(baseurl, params)
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
                settings[setting] = trmat
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
    params = urllib.urlencode(params)
    #print params
    result = urllib.urlopen(url, params)
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
        generator = map(sp.S, pre.split())
        generators.append(sp.Matrix(generator).reshape(3,4))
    return generators
    

def get_generators(sgnum=0, sgsym=None):
    """
        Retrieves all Generators of a given Space Group Number (sgnum) a local
        sqlite database OR from http://www.cryst.ehu.es and stores them into
        the local sqlite database.
        
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
        settings = np.genfromtxt(SETTPATH, dtype="i2,O,O")
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
        setnames = " ".join(settings.keys())
        if len(settings)==1:
            sgsym = settings.keys()[0]
        elif sgsym!=None:
            if sgsym not in settings:
                raise ValueError(
                 "Invalid space group symbol (sgsym) entered: %s%s"\
                 "  Valid space group settings: %s"\
                 %(sgsym, os.linesep,setnames))
        else:
            print("Warning: Space Group #%i is ambigous:%s"\
                  "  Possible settings: %s "%(sgnum, os.linesep, setnames))
            settings_inv = dict([(v,k) for (k,v) in settings.iteritems()])
            sgsym = settings_inv["a,b,c"]
            print("  Using standard setting: %s"%sgsym)
        trmat = settings[sgsym]
    else:
        raise ValueError("Integer required for space group number (`sgnum')")
    
    import sqlite3
    import base64
    dbi=sqlite3.connect(DBPATH)
    cur=dbi.cursor()
    cur.execute("SELECT * FROM spacegroups WHERE sg_symbol = '%s'" %sgsym)
    result=cur.fetchone()
    dbi.close()
    if result:
        try:
            return pickle.loads(base64.b64decode(result[2]))
        except:
            dbi=sqlite3.connect(DBPATH)
            cur=dbi.cursor()
            cur.execute("DELETE FROM spacegroups WHERE sg_symbol = '%s'"%sgsym)
            dbi.commit()
            dbi.close()
            print "Deleted old dataset."
    print("Space Group %s not yet in database. "\
          "Fetching from internet..."%sgsym)
    print sgnum, trmat
    generators = fetch_ITA_generators(sgnum, trmat)
    gendump = base64.b64encode(pickle.dumps(generators))
    dbi=sqlite3.connect(DBPATH)
    cur=dbi.cursor()
    cur.execute("INSERT INTO spacegroups "\
                  "(sg_symbol, sg_number, generators, trmat) "\
                "VALUES "\
                  "('%s', '%i', '%s', '%s')"\
                 % (sgsym, sgnum, gendump, trmat))
    dbi.commit()
    dbi.close()
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

def makefunc(expr, mathmodule = "numpy"):
    symbols = list(expr.atoms(sp.Symbol))
    symbols.sort(key=str)
    func = lambdify(symbols, expr, mathmodule, dummify=False)
    func.kw = symbols
    func.expr = expr
    func.kwstr = map(lambda x: x.name, symbols)
    func.dictcall = types.MethodType(dictcall, func)
    func.__doc__ = str(expr)
    return func

class makeufunc(object):
    def __init__(self, expr):
        symbols = list(expr.atoms(sp.Symbol))
        symbols.sort(key=str)
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
    for i in xrange(Tensor.ndim):
        Axes = range(Tensor.ndim)
        Axes[0] = i
        Axes[i] = 0
        Tensor = numpy.tensordot(Matrix, Tensor.transpose(Axes), axes=1).transpose(Axes)
    return Tensor

def full_transform2(Matrix, Tensor):
    """
        Transforms the Tensor to Representation in new Basis with given Transformation Matrix.
    """
    for i in xrange(Tensor.ndim):
        Tensor = np.tensordot(Tensor, Matrix, axes=(0,0))
    return Tensor

def full_transform(Matrix, Tensor):
    """
        Transforms the Tensor to Representation in new Basis with given Transformation Matrix.
    """
    Matrix = np.array(Matrix)
    Tensor = np.array(Tensor)
    dtype = np.find_common_type([],[Matrix.dtype, Tensor.dtype])
    Tnew = np.zeros_like(Tensor, dtype = dtype)
    for ind in itertools.product(*map(range, Tensor.shape)):
        for inds in itertools.product(*map(range, Tensor.shape)):
            Tnew[ind] += Tensor[inds] * Matrix[inds, ind].prod()
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
        print system
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
