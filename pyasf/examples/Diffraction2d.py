from pylab import *
import pyasf
gamma = 1
bg = 1e-12

det = pyasf.AreaDetector((22, 0, 0.2)) # delta, nu, dist
energy = [28000] # eV
alpha = 15.7 # angle of incidence
psi = 0 # azimuth
cs = pyasf.unit_cell("7101739") # from crystallography open database
s = pyasf.Geometry.ThreeCircleVertical(cs, (1,1,1))

fig, ax = subplots(1,3)
ax = ax.ravel()

Q,im = s.Diffraction2D(energy, alpha, det, psi, divergence=(1,1),  verbose=True, dEvsE=1e-4)

im = im**gamma

ax[0].imshow(flipud(log10(im+bg)))#, vmax=1000.**gamma)
ax[0].set_title("$I_\\mathrm{max}=%g$"%(im.max()))

ax[1].pcolormesh(Q[0],Q[1],log10(im+bg))
ax[1].set_xlabel("h")
ax[1].set_ylabel("k")

ax[2].pcolormesh(Q[0],Q[2],log10(im+bg))
ax[2].set_xlabel("h")
ax[2].set_ylabel("l")
show()