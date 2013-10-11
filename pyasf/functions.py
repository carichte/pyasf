import os
import sympy as sp
import pyxrr
DBPATH = os.path.join(os.path.split(__file__)[0], "space-groups.sqlite")
#expanduser("~/Arbeit/Tabellen/space-groups.sqlite")


def get_generators(sg):
    """
        Retrieves all Generators of a given Space Group (sg) a local sqlite database
        OR from http://www.cryst.ehu.es and stores them into the local sqlite database.
    """
    import sqlite3
    import urllib
    import base64
    import pickle
    dbi=sqlite3.connect(DBPATH)
    cur=dbi.cursor()
    cur.execute("SELECT * FROM spacegroups WHERE spacegroup = '%i'" %int(sg))
    result=cur.fetchone()
    dbi.close()
    if result:
        try:
            return pickle.loads(base64.b64decode(result[1]))
        except:
            dbi=sqlite3.connect(DBPATH)
            cur=dbi.cursor()
            cur.execute("DELETE FROM spacegroups WHERE spacegroup = '%i'" %int(sg))
            dbi.commit()
            dbi.close()
            print "Deleted old dataset."
    print "Space Group " + str(sg) + " not yet in database. Fetching from internet..."
    generators=[]
    params = urllib.urlencode({'gnum': sg, 'what':'gp'})
    result=urllib.urlopen("http://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-getgen", params)
    result=result.read()
    while "<pre>" in result:
        smth=result.partition("<pre>")[2].partition("</pre>")
        generator=map(sp.S, smth[0].split())
        print generator
        generators.append(sp.Matrix(generator).reshape(3,4))
        result=smth[2]
    dbi=sqlite3.connect(DBPATH)
    cur=dbi.cursor()
    cur.execute("INSERT INTO spacegroups (spacegroup, generators) VALUES  ('%i', '%s')" % (sg, base64.b64encode(pickle.dumps(generators))))
    dbi.commit()
    dbi.close()
    return generators
    

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
        else: raise ValueError("Invalid Space Group Number. Has to be in range(1,231).")
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
    
    B = sp.Matrix([[ar, br*sp.cos(gammar),  cr*sp.cos(betar)],
                   [0,  br*sp.sin(gammar), -cr*sp.sin(betar)*sp.cos(alpha)],
                   [0,  0,                  1/c]])
    #V_0 = sp.sqrt(1 - sp.cos(alphar)**2 - sp.cos(betar)**2 - sp.cos(gammar)**2 + 2*sp.cos(alphar)*sp.cos(betar)*sp.cos(gammar))
    #print V_0
    
    B_0 = sp.Matrix([[1, sp.cos(gammar),  sp.cos(betar)],
                     [0, sp.sin(gammar), -sp.sin(betar)*sp.cos(alpha)],
                     [0, 0,               1]])
    
    
    
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
    
    

