import os
import sympy as sp
import pyxrr
import urllib
import pickle
DBPATH = os.path.join(os.path.dirname(__file__), "space-groups.sqlite")
SETTPATH = os.path.join(os.path.dirname(__file__), "settings.dump")
GENPATH =  os.path.join(os.path.dirname(__file__), "generators.dump")


def get_ITA_settings(sgnum):
    if os.path.isfile(SETTPATH):
        with open(SETTPATH, "r") as fh:
            transformations = pickle.load(fh)
    else:
        transformations = {}
    
    if sgnum in transformations:
        settings = transformations[sgnum]
    else:
        from lxml import etree
        print("Fetching ITA settings for space group %i"%sgnum)
        baseurl = "http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-getgen"
        params = urllib.urlencode({'gnum': sgnum, 'what':'gp', 'settings':'ITA Settings'})
        result=urllib.urlopen(baseurl, params)
        result=result.read()
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
        transformations[sgnum] = settings
        with open(SETTPATH, "w") as fh:
            pickle.dump(transformations, fh)
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
    for tr in list(table) + list(table.find("tbody")):
        if not isinstance(tr[0].text, str) or not tr[0].text.isdigit():
            continue
        #return tr
        pre = tr[4][0][0][1][0].text
        generator = map(sp.S, pre.split())
        generators.append(sp.Matrix(generator).reshape(3,4))
    return generators
    

def get_generators(sgnum, sgsym=None):
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
    if isinstance(sgnum, int):
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
    else:
        raise ValueError("Integer required for space group number (`sgnum')")
    trmat = settings[sgsym]
    
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
    if [i.is_Symbol for i in coordinate.atoms()].count(True) > 0: return coordinate
    else: return coordinate%1
    

@sp.vectorize(0)
def hassymb(x):
    return [i.is_Symbol for i in x.atoms()].count(True) > 0

def full_transform(Matrix, Tensor):
    """
        Transforms the Tensor to Representation in new Basis with given Transformation Matrix.
    """
    import numpy
    for i in range(len(Tensor.shape)):
        Axes = range(len(Tensor.shape))
        Axes[0] = i
        Axes[i] = 0
        Tensor = numpy.tensordot(Matrix, Tensor.transpose(Axes), axes=1).transpose(Axes)
    return Tensor

def get_cell_parameters(sg):
        """
            Returns the general cell parameters for a lattice of given space group number sg.
        """
        import sympy as sp
        a, b, c, alpha, beta, gamma = sp.symbols("a, b, c, alpha, beta, gamma",
                                                 commutative=True,
                                                 complex=True,
                                                 imaginary=False,
                                                 negative=False,
                                                 nonnegative=True,
                                                 real=True,
                                                 bounded=True,
                                                 positive=True,
                                                 unbounded=False)
        
        if sg in range(1,3): 
            print "Triclinic"
        elif sg in range(3,16):
            alpha=sp.S("pi/2")
            gamma=sp.S("pi/2")
            print "Monoclinic"
        elif sg in range(16,75): 
            alpha=sp.S("pi/2")
            beta=sp.S("pi/2")
            gamma=sp.S("pi/2")
            print "Orthorhombic"
        elif sg in range(75,143):
            b=a
            alpha=sp.S("pi/2")
            beta=sp.S("pi/2")
            gamma=sp.S("pi/2")
            print "Tetragonal"
        elif sg in range(143,168):
            b=a
            alpha=sp.S("pi/2")
            beta=sp.S("pi/2")
            gamma=sp.S("2/3*pi")
            print "Trigonal (hexagonal setting)"
        elif sg in range(168,195):
            b=a
            alpha=sp.S("pi/2")
            beta=sp.S("pi/2")
            gamma=sp.S("2/3*pi")
            print "Hexagonal"
        elif sg in range(195,231):
            b=a
            c=a
            alpha=sp.S("pi/2")
            beta=sp.S("pi/2")
            gamma=sp.S("pi/2")
            print "Cubic"
        else: raise ValueError(
                    "Invalid Space Group Number. Has to be in range(1,231).")
        return a, b, c, alpha, beta, gamma

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
    
    # volume of crystall system cell in carthesian system
    # V = sp.sqrt(G.det())
    
    # cell vectors in carthesian system in reciprocal space
    #ar_vector = 2*sp.pi* cross(b_vector, c_vector) / V
    #br_vector = 2*sp.pi* cross(c_vector, a_vector) / V
    #cr_vector = 2*sp.pi* cross(a_vector, b_vector) / V
    
    # reciprocal cell lengths
    ar = sp.sqrt(G_r[0,0])
    br = sp.sqrt(G_r[1,1])
    cr = sp.sqrt(G_r[2,2])
    
    alphar = sp.acos(G_r[1,2]/(br*cr))
    betar = sp.acos(G_r[0,2]/(ar*cr))
    gammar = sp.acos(G_r[0,1]/(ar*br))
    
    B =   sp.Matrix([[ar, br*sp.cos(gammar),  cr*sp.cos(betar)],
                     [0,  br*sp.sin(gammar), -cr*sp.sin(betar)*sp.cos(alpha)],
                     [0,  0,                  1/c]])
    #V_0 = sp.sqrt(1 - sp.cos(alphar)**2 - sp.cos(betar)**2 - sp.cos(gammar)**2 + 2*sp.cos(alphar)*sp.cos(betar)*sp.cos(gammar))
    #print V_0
    
    B_0 = sp.Matrix([[1,  sp.cos(gammar),     sp.cos(betar)],
                     [0,  sp.sin(gammar),    -sp.sin(betar)*sp.cos(alpha)],
                     [0,  0,                  1]])
    
    
    
    return (ar, br, cr, alphar, betar, gammar, B, B_0, G, G_r)

def calc_f0(element, q, database = pyxrr.DB_PATH): # q is 2 sin(theta)/lambda
    """
        Calculates the nonresonant scattering factor for elements and common ions
        in the range of sin(theta)/lambda < 2.0 per Angstrom.
        
        The input q corresponds to 2*sin(theta)/lambda and can be a symbolic value.
    """
    x = q/2.
    if (x>2)==True:
        print("Warning: the used expansion only gives a good agreement for values "\
               "sin(theta)/lambda < 2 A^-1. The value entered (%g) exceeds this"%float(x))
    import sqlite3
    import sympy
    dbi = sqlite3.connect(database)
    cur = dbi.cursor()
    cur.execute("SELECT * FROM f0_lowq WHERE ion = '%s'" %element)
    result=cur.fetchone()
    dbi.close()
    a = sympy.Matrix(result[1:5])
    b = sympy.Matrix(result[5:9])
    c = result[9]
    #return  (a,(-b*(q/2)**2).applyfunc(sympy.exp)) # + c
    f0 = sum(a.multiply_elementwise((-b*x**2).applyfunc(sympy.exp))) + c
    return f0
    
    

