import os
import pyasf
import numpy as np
import sympy as sp
import itertools
        

class ThreeCircleVertical(object):
    """
        A Single Crystal diffraction Geometry in vertical position:
        
        * Orientation || z fixed:
        * Azimuthal position `psi` variable
        
        During diffraction:
        * Bragg condition via incidence angle `alpha`
            - no eta (omega)
            - no chi
        * `delta` variable
        * `nu` variable
    """
    _cache = {}
    _debug = {}
    _geomparam = {}
    def __init__(self, structure, orientation, psi=0):
        """
            Define Single crystal orientation:
            
            Inputs:
                structure : a <pyasf.unit_cell> object
                    defines the crystal structure
                
                orientation : a 3-tuple
                    miller indizes defining the reciprocal
                    lattice vector along z
                
                psi : float
                    the rotation around that vector in deg
        """
        self.structure=structure
        self.orientation=orientation
        self.calc_orientation_from_angle(orientation, psi)
    
    def calc_orientation_from_angle(self, orientation, psi=0):
        """
            Calculates matrix for transformation from cartesian
            crystal-fixed system to laboratory system (`R` or `U`).
            
            Inputs:
                orientation : a 3-tuple
                    miller indizes defining the reciprocal
                    lattice vector along z
                
                psi : float
                    the rotation around that vector in deg
        """
        psi = sp.rad(psi).n()
        self.orientation = orientation
        orientation = sp.Matrix(orientation)
        M = self.structure.M
        w3 = M.T.inv()*orientation/((M.T.inv()*orientation).norm())
        b_rec = sp.Matrix([0,1,0])
        b = M.T.inv()*b_rec
        pv = b.cross(w3)
        if pv.norm()==0:
            c_rec=sp.Matrix([0,0,1])
            c = M.T.inv()*c_rec
            pv = w3.cross(c)
        p = pv.normalized()
        q = p.cross(w3)
        w1 = p*sp.cos(psi)+q*sp.sin(psi)
        v_par = M*w1
        w2 = w3.cross(w1)
        R = sp.Matrix([w1.T, w2.T, w3.T])
        R.simplify()
        self._geomparam["w1"] = w1
        self._geomparam["w3"] = w3
        self._geomparam["v_par"] = v_par
        self.R = R
        return R
    
    
    def Diffraction2D(self, energy, alpha, detector, psi=None, dEvsE=1e-4,
                      divergence=(1,1), verbose=True, **kwargs):
        """
            Calculation of 2d detector cut in reciprocal space.
            
            Inputs:
                energy : float / 1darray (eV)
                    the x-ray energies coming from the source
                alpha : float (deg)
                    the angle between the incident beam and the x-axis
                    or the orientation lattice planes
                detector : <AreaDetector> class instance
                    contains the detector geometry
                psi : float (deg)
                    the rotation around that vector in deg
                dEvsE : float
                    the relative energy resolution
                divergence : float, 1- or 2-tuple (mrad)
                    defining the divergence/convergence of the 
                    incident beam in mrad
                    -> angular deviation (sigma) from the central beam
            
            Other inputs:
                some keyword arguments are forwarded to the routines
                `calc_structure_factor` and `DAFS` of the 
                pyasf.unit_cell class.
            
            Returns:
                Q : 2darray
                    the value of momentum transfer in relative
                    lattice units for each detector pixel
                    (|Q| = 2/lambda sin(theta))
                I : 2darray
                    the total intensity in fractions of the primary beam
                    (I<1) for divergence = 1mrad
        """
        debug = kwargs.get("debug", False)
        
        SF_defaults = dict(DD=False, 
                           DQ=False, 
                           subs=True, 
                           evaluate=True,
                           subs_U=True,
                           Temp=True,
                           Uaniso=bool(len(self.structure.Uaniso)),
                           simplify=False)
        
        SF_defaults.update([kw for kw in kwargs.iteritems() if kw[0] in SF_defaults])
        
        DAFS_kw = ("DD", "DQ", "Temp", "fwhm_ev", "table", "simplify", "subs", "Uaniso")
        DAFS_kw = dict([(k,v) for k in kwargs.iteritems() if k in DAFS_kw])
        
        AAS = SF_defaults["DD"] or SF_defaults["DQ"]
        if verbose:
            print SF_defaults
            #print("Calculating all structure factors...")
        
        
        energy = np.array(energy, dtype=float, ndmin=1)
        #alpha = np.degrees(alpha)
        alpha = sp.rad(alpha).n()
        var = (np.array(divergence, dtype=float, ndmin=1)/1000.)**2
        if len(var)==1:
            var = var.repeat(2) #radians
        
        assert energy.ndim==1, \
            "Wrong shape of array of energies. Only 0D or 1D supported"
        assert (var>0).all(), \
            "divergence must be finite"
        
        structure = self.structure
        
        if psi is None:
            R = self.R # rotation matrix, also called U in literature
        else:
            if verbose:
                print("Rotating sample...")
            miller = self.orientation
            R = self.calc_orientation_from_angle(self.orientation, psi)
        
        M = structure.M.subs(structure.subs) # also called B in literature
        R = R.subs(structure.subs).n()
        RM = R*M
        RMi = RM.inv()
        RM.simplify()
        RMi.simplify()
        
        if verbose:
            print os.linesep,"rec. lattice to cartesian coordinates:"
            sp.pprint(M.inv().T.n(3))
        if verbose:
            print os.linesep,"cartesian to diffractometer coordinates:"
            sp.pprint(R.n(3))
        
        structure.calc_structure_factor(**SF_defaults)
        # forward scattering amplitude:
        F_0 = structure.DAFS(energy, (0,0,0), force_refresh=False, **DAFS_kw)
        # a symbol for forward scattering:
        F_0_sym = sp.Symbol("F_0", complex=True) 
        
        # direction of incident wavevector:
        vec_k_i = sp.Matrix([sp.cos(alpha), 0, -sp.sin(alpha)])
        vec_k_i_ = np.array(vec_k_i)
        if verbose:
            print os.linesep,"incident X-ray wavevector:"
            sp.pprint(vec_k_i.n(4))
        
        # unit wave vector for reflection h (K(h_m)), symbolic:
        ksx, ksy, ksz = sp.symbols("k_sx k_sy k_sz", real=True)
        vec_k_s = sp.Matrix([ksx, ksy, ksz])
        
        
        
        k = structure.energy / structure.eV_A # scalar wavevector transfer k=1/lambda
        Q = (vec_k_s - vec_k_i)*k # unit wave vector transfer
        Qcryst = RM.T * Q # wave vector transfer in crystal units
        
        self.Qfunc = Qfunc = pyasf.makefunc(Qcryst)
        kval = detector.get_kprime()
        
        # determine all possible rec. space vectors:
        #Qval = Qfunc(np.array([energy.min()/1.1, energy.max()*1.1])[:,None,None], *kval)[:,0]
        Qval = Qfunc(energy[:,None,None], *kval)[:,0]
        if debug:
            self._debug["Qval"] = Qval
        
        # these are the corners of the cube through which we have to 
        # integrate in reciprocal space
        Hmin = np.floor(Qval.min((1,2,3))).astype(int)
        Hmax =  np.ceil(Qval.max((1,2,3))).astype(int)
        #Qval = Qfunc(energy[:,None,None], *kval)[:,0]
        if verbose:
            print("Integration in reciprocal space from %i %i %i"
                  " to %i %i %i"%(tuple(Hmin)+tuple(Hmax)))
        
        
        
        subs = [(sym, structure.subs[sym]) for sym in structure._metrik if sym in structure.subs]
        V = structure.V.subs(subs)
        r_e = structure.electron_radius*1e10 # 2.818e-5 in Angstrom
        Gamma = r_e / (sp.pi * k**2 * V)
        Gamma_ = pyasf.makeufunc(Gamma)

        Hcart = structure.Gc.subs(subs)
        Hlab = R * Hcart
        k0 = vec_k_s - Hlab/k # shorten vec_k_s according to refraction?
        normk0 = k0.dot(k0)
        
        # the length of this vector divided by sqrt(normk0) is the sine 
        # of the angular deviation of the incident wavevector:
        div = vec_k_i.cross(k0)
        # decomposition into vertical and horizontal part:
        div_v = (div[1]**2/normk0).simplify()
        div_v_f = pyasf.makeufunc(div_v)
        div_h = ((div[0]**2 + div[2]**2)/normk0).simplify()
        div_h_f = pyasf.makeufunc(div_h)
        if debug:
            self._debug["div_h_f"] = div_h_f
            self._debug["div_v_f"] = div_v_f
        
        # the excitation error: uncertainty along \vec k
        # without broadening:
        #R_h = 1. / normk0 - (1 + Gamma*F_0_sym) # symbolic, -Gamma*F(0) = chi(0)
        """
         With fast broadening close to uniform:
            these functions are the envelope of a set of
            y=x that corresponds to the value closest to 1.
            This way the excitation error will be maximum.
        """
        broaden = lambda x: (x - 2/sp.pi*sp.atan((x-1)/dEvsE*     sp.pi /2)*dEvsE) # lorentzian
        #broaden = lambda x: (x -     special.erf((x-1)/dEvsE*sqrt(sp.pi)/2)*dEvsE) # gaussian
        R_h = 1. /broaden(normk0) - (1 + Gamma*F_0_sym)
        self.R_h_f = R_h_f = pyasf.makefunc(R_h, "numpy")
        
        if debug:
            self._debug["k0"] = k0
            self._debug["normk0"] = normk0
            self._debug["R_h_f"] = R_h_f
        
        self.subs = subs = {structure.energy:energy[:,None,None], 
                            ksx:kval[0], 
                            ksy:kval[1], 
                            ksz:kval[2], 
                            F_0_sym:F_0[:,None,None]}
                    


        indices = [np.arange(Hmin[i], Hmax[i]+1) for i in xrange(3)]
        riter = itertools.product(*map(tuple,indices))
        im = np.zeros((len(energy),)+kval.shape[1:])
        Gamma_ = Gamma_.dictcall(subs)
        #for H in structure.iter_rec_space(qmax=2, independent=False): #all reflections?
        for H in riter: # iterate over the cube in Q space
            subs.update(zip(structure.miller, H))
            F = structure.DAFS(energy, H, force_refresh=False)[:,None,None]
            if abs(F).max() < 1e-4: # discard weak reflections
                continue
            
            if debug:
                print(subs)
            
            angh = div_h_f.dictcall(subs) # horizontal divergence
            if (angh.min()/(2*var[0])) > 10:
                continue
            angv = div_v_f.dictcall(subs) # vertical divergence
            if (angv.min()/(2*var[1])) > 10:
                continue
            
            
            R_h = 1./R_h_f.dictcall(subs)
            
            
            I = abs((Gamma_ * F * R_h))**2 \
               * np.exp(- angh/(2*var[0]) \
                        - angv/(2*var[1]))/np.sqrt((var[0]*var[1]))/1000**2
            if verbose:
                print H,
                if I.max()>1e-10:
                    imax = I.argmax()
                    print("Excitation error: %g"%(1./abs(R_h).max())),
                    print("Peak intensity: %g"%I.max()),
                    print("h,k,l: %.3f %.3f %.3f"%tuple(Qval[i].ravel()[imax] for i in xrange(3))),
                print("")
            
            im += I # uniform sum over energies
        
        return Qval.squeeze(), (im*detector.get_projection()).squeeze()


