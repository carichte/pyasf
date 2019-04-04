import numpy as np


class AreaDetector(object):
    """
        Class defining geometry of 2D (area) detectors
        
        Simplified geometry:
            point of normal incidence assumed to be in the center pixel
        
    """
    def __init__(self, position, pixelNum=1024, pixelsize=50e-6):
        """
            Create new detector instance
            
            Inputs:
                position : sequence of floats of length 2 or 3
                    defines the position of the center of the detector
                    by polar (delta, distance) 
                    or spherical (delta, nu, distance) coordinates
                    
                    units: delta, nu (deg)
                           distance (m)
                pixelNum : int or 2-tuple of int (rows * cols)
                    the number of pixels in both directions
                
                pixelsize : float (m)
                    the edge length of the pixels
        """
        if isinstance(pixelNum, int):
            pixelNum = (pixelNum, pixelNum) #rows*cols
        
        if len(position)==2: #only given as angle and distance
            delta, sdd = position
            delta = np.radians(delta)
            poni = np.array([
                    np.cos(delta),
                    0,
                    np.sin(delta)]) * sdd
        elif len(position)==3:
            delta, nu, sdd = position
            delta = np.radians(delta)
            nu    = np.radians(nu)
            poni = np.array([
                    np.cos(nu)*np.cos(delta),
                    np.sin(nu)*np.cos(delta),
                    np.sin(delta)]) * sdd
        else:
            raise ValueError
        
        self.poni = np.array(poni, dtype=float)
        self.pixelNum = pixelNum
        self.pixelsize = pixelsize
        if hasattr(self, "projection"):
            del self.projection
        if hasattr(self, "kprime"):
            del self.kprime
    
    def get_pixels(self):
        rows, cols = self.pixelNum
        rows = np.arange(-rows//2, rows//2)
        cols = np.arange(-cols//2, cols//2)
        cols, rows = np.meshgrid(cols, rows)
        return rows, cols
    
    def get_kprime(self):
        vhoriz = np.cross([0,0,1], self.poni)
        vhoriz /= np.linalg.norm(vhoriz)
        vverti = np.cross(self.poni, vhoriz)
        vverti /= np.linalg.norm(vverti)
        
        rows, cols = self.get_pixels()
        rows = rows * self.pixelsize
        cols = cols * self.pixelsize
        
        kprime = self.poni[:,None,None] + \
                 vhoriz[:,None,None]*cols + \
                 vverti[:,None,None]*rows
        
        self.kprime = kprime = kprime/np.linalg.norm(kprime, axis=0)
        return kprime
    
    def get_projection(self):
        if hasattr(self, "projection"):
            return self.projection
        kprime = self.kprime if hasattr(self, "kprime") else self.get_kprime()
        
        kcen = kprime[:, self.pixelNum[0]//2, self.pixelNum[1]//2]
        projection = np.tensordot(kprime, kcen, axes=(0,0))
        return projection
        

