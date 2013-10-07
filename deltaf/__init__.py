import numpy as np
import os
import elements
import deltaf
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import quad
from scipy.optimize import fmin


__doc__ =  """
        This is a tiny wrapper for Matt Newville's deltaf Fortran program.
        It returns the correction terms f' and f'' of the atomic scattering
        factors for a given element and energy range.
        
        See deltaf.getf
    """

def cauchy(x, w):
    return 1./np.pi * w / (w**2 + x**2)

def get_transition(element, transition=None, col="Direct"):
    """
        Returns all x-ray absorption edge and line enegies for a
        given element. Optionally a certain transition can be chosen.
        
        Inputs:
            element : str
                exact symbol of the chemical element of interest.
        
        Optional inputs:
            transition : str
                name of selected transition
                possible values are:
                    ['K edge',
                     'KL1', 'KL2', 'KL3',
                     'KM1', 'KM2', 'KM3', 'KM4', 'KM5', 'KN1',
                     'L1 edge',
                     'L1M1', 'L1M2', 'L1M3', 'L1M4', 'L1N1',
                     'L2 edge', 'L2M1', 'L2M2', 'L2M3', 'L2M4', 'L2N1',
                     'L3 edge', 'L3M1', 'L3M2', 'L3M3', 'L3M4', 'L3M5', 'L3N1']
            
            col : str
                selected dataset.
                can be one of:
                    ["Theory", "TheoryUnc",
                     "Direct", "DirectUnc",
                     "Combined", "CombinedUnc",
                     "Vapor",  "VaporUnc"]
                where Unc denotes uncertainties of the values
        
        The data was obtained from (see for more information):
            Deslattes R D, Kessler E G, Indelicato P, de Billy L, Lindroth E and Anton J,
            Rev. Mod. Phys. 75, 35-99 (2003)
            10.1103/RevModPhys.75.35
            """
    DB_NAME = "transition_emission_energies.npz"
    DB_PATH = os.path.join(os.path.dirname(__file__), DB_NAME)
    try: Z = int(element)
    except: Z = elements.Z[element]
    db = np.load(DB_PATH)
    try:
        ind = db["Z"]==Z
        transitions = db["type"][ind]
        energies = db[col][ind]
    except Exception as errmsg:
        print errmsg
    finally:
        db.close()
    ind = np.where(transitions==transition)[0]
    if len(ind):
        return energies[ind]
    else:
        return dict(zip(transitions, energies))
    

def get_edge(element, edge=None):
    """
        Returns x-ray absorption edge of a chemical element of choice.
        A particular absorption edge can be chosen out of
            {Z, A, K, L1, L2, L3, M1, M5},
        where Z and A denote atomic number and atomic weight, respectively.
    """
    DB_NAME = "elements.npz"
    DB_PATH = os.path.join(os.path.dirname(__file__), DB_NAME)
    try: Z = int(element)
    except: Z = elements.Z[element]
    db = np.load(DB_PATH)
    cols = db.files
    try:
        ind = Z-1
        edges = dict([(col, db[col][ind]) for col in cols])
    except Exception as errmsg:
        print errmsg
    finally:
        db.close()
    if edge in cols:
        return edges[edge]
    else:
        return edges

def getf(element, energy, conv_fwhm=0):
    """
        This is a tiny wrapper for Matt Newville's deltaf Fortran program.
        It returns the correction terms f' and f'' of the atomic scattering
        factors for a given element and energy range.
        
        Inputs:
            - element : int or str - atomic number or abbreviation of element
            - energy  : array of energies in eV
            - conv_fwhm : float - width of lorentzian for broadening in eV.
        
        Outputs:
            f1, f2 Tuple
    """
    try: Z = int(element)
    except: Z = elements.Z[element]
    diffE = np.unique(np.diff(energy))
    dE = abs(diffE.min())
    fwhm_chan = abs(float(conv_fwhm))/dE
    # make equally spaced data:
    if diffE.std()/diffE.mean() > 1e-5 and fwhm_chan>1e-2:
        newenergy = np.arange(energy.min() - 10*dE, energy.max() + 10.*dE, dE)
        do_interpolate = True
    else:
        newenergy = energy
        do_interpolate = False
        
    f1, f2 = deltaf.clcalc(Z,  newenergy)
    if fwhm_chan>1e-2:
        f1, f2 = deltaf.convl2(f1, f2, fwhm_chan)
    if do_interpolate:
        ffunc = interp1d(newenergy, (f1, f2))
        f1, f2 = ffunc(energy)
    return f1, f2
    

def getfquad(element, energy=None, fwhm_ev = 0, lw = 50, f1f2 = "f1"):
    try: Z = int(element)
    except: Z = elements.Z[element]
    
    if energy==None:
        energy, iedge = get_energies(Z, 1000, 10000, fwhm_ev=fwhm_ev)
        return_ene = True
    else:
        return_ene = False
    
    if f1f2 == "f2" or f1f2 == 1:
        ind = 1
    else:
        ind = 0
    
    fwhm_ev = abs(fwhm_ev)
    
    if fwhm_ev <= np.finfo(float).eps:
        result =  deltaf.clcalc(Z, energy)[ind]
    else:
        
        corrfac = 1./quad(cauchy, -lw, lw, args=(1,), limit=500)[0]
        
        integrand = lambda x, E: cauchy(x, fwhm_ev) * deltaf.clcalc(Z, E-x)[ind]
            
        def ffunc(E):
            return quad(integrand, -lw*fwhm_ev, lw*fwhm_ev, args=(E,), limit=500)[0]
        
        if np.isscalar(energy) or len(energy)==1:
            result = corrfac * ffunc(energy)
        else:
            fvec = np.vectorize(ffunc)
            result = corrfac * fvec(energy)
    
    if return_ene:
        return energy, result
    else:
        return result



def get_energies(element, emin, emax, fwhm_ev=1e-4, eedge = None, num=100):
    assert emax>emin, "emax must be larger than emin."
    fwhm_ev = max(abs(fwhm_ev), 1e-6)
    try: Z = int(element)
    except: Z = elements.Z[element]
    def f1func(E):
        return deltaf.clcalc(Z, E)[0]
    if eedge == None:
        eedge = .5 * (emin + emax)
    eedge = float(fmin(f1func, (eedge,)))
    print "Found edge at %g"%eedge
    expmin = np.floor(np.log10(fwhm_ev))
    dist = float(2*max(abs(emax - eedge), abs(emin - eedge)))
    expmax = np.log10(dist/2.)
    expmin, expmax = min(expmin, expmax), max(expmin, expmax)
    ewing = np.logspace(expmin, expmax, num)
    dE = ewing[1] - ewing[0]
    num_lin = 2*int(ewing[0] / dE) + 1
    ecenter = np.linspace(-ewing[0], ewing[0], num_lin)[1:-1]
    energy = np.append(-ewing[::-1], ecenter)
    energy = np.append(energy, ewing)
    energy += eedge
    energy = energy[energy>0]
    try:
        iedge = float(np.where(energy==eedge)[0])
    except:
        raise ValueError("Edge energy not found in constructed array")
    return energy, iedge
