import analytical_potential as ap
import grid
import matplotlib.pyplot as plt
import multipole
import numpy as np
import L2_difference as L2
import comparitive_difference as comdf

# test with 16*16*16, 32*32*32 and 64*64*64

nx = 16
ny = 16
nz = 16
'''
nx = 32
ny = 32
nz = 32
'''

nx = 64
ny = 64
nz = 64

'''
nx = 128
ny = 128
nz = 128
'''

xlim = (-0.5, 0.5)
ylim = (-0.5, 0.5)
zlim = (-0.5, 0.5)
g = grid.Grid(nx, ny, nz, xlim, ylim, zlim)
dens = g.scratch_array()
'''
# density of a perfect sphere
dens = g.scratch_array()
sph_center = (0.0, 0.0, 0.0)
radius = np.sqrt((g.x3d - sph_center[0])**2 + (g.y3d - sph_center[1])**2 +
                 (g.z3d - sph_center[2])**2)

# a = 0.25
a = 0.25
print("a = ", a)
mask = radius <= a
dens[mask] = 1.0
density = 1.0

# normalized density of the sphere
m = 4/3*np.pi*a**3*density
# actural volume of a sphere
mask_mass = mask
vol_mass = np.sum(g.vol[mask_mass])
density_norm = m/vol_mass
dens_norm = g.scratch_array()
dens_norm[mask_mass] = density_norm
'''
'''
# analytical potential of a sphere at a y plane
phi_anal = g.scratch_y_plane_array()
for i in range(g.nx):
    for j in range(g.nz):
        sph_phi = ap.Ana_Sph_pot(g.x[i], g.y[8], g.z[j], a, density)
        phi_anal[i, j] = sph_phi.potential
        '''
'''
xlim = (-1.0, 1.0)
ylim = (-1.0, 1.0)
zlim = (-1.0, 1.0)
g = grid.Grid(nx, ny, nz, xlim, ylim, zlim)
dens = g.scratch_array()

# density of the cube
a = 0.5
b = 0.5
c = 0.5
sites = (a, b, c)
# density of a cube
for i in range(g.nx):
    for j in range(g.ny):
        for k in range(g.nz):
            if g.x[i] >= 0 and \
                    g.x[i] <= a and \
                    g.y[j] >= 0 and \
                    g.y[j] <= b and \
                    g.z[k] >= 0 and \
                    g.z[k] <= c:
                dens[i, j, k] = 1.0

# analytical potential of a cube
phi_anal = g.scratch_y_plane_array()
for i in range(g.nx):
    for j in range(g.nz):
        cube_V = ap.Analytical_Cube_Potential(sites, g.x[i], g.y[10], g.z[j])
        phi_anal[i, j] = cube_V.V
        '''

# test the spheroid
# density of the spheroid
a3 = 0.10
e = 0.9
a1 = a3/np.sqrt(1-e**2)
mask = g.x3d**2/a1**2 + g.y3d**2/a1**2 + g.z3d**2/a3**2 <= 1
dens[mask] = 1.0
density = 1.0

# normalized density of a spheroid
m = 4/3*np.pi*a1**2*a3*density
# actural volume of a sphere
mask_mass = mask
vol_mass = np.sum(g.vol[mask_mass])
density_norm = m/vol_mass
dens_norm = g.scratch_array()
dens_norm[mask_mass] = density_norm

'''
# analytical potential of a spheroid at y surface
phi_anal = g.scratch_y_plane_array()
for i in range(g.nx):
    for j in range(g.nz):
        spheroid_V = ap.Ana_Mac_pot(a1, a3, g.x[i], g.y[8], g.z[j], density)
        phi_anal[i, j] = spheroid_V.potential
        '''

lmax = 30
center = (0, 0, 0)
# m = multipole.Multipole(g, dens, lmax, 0.3*g.dr, center=center)
m = multipole.Multipole(g, dens_norm, lmax, 0.3*g.dr, center=center)
phi = m.Phi()
'''
phi = m.PhiY(8)

plt.imshow(np.log10(np.abs(np.transpose(phi))), origin="lower",
           interpolation="nearest",
           extent=[g.xlim[0], g.xlim[1],
                   g.zlim[0], g.zlim[1]])

plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("lmax=0.png")

diff = comdf.comp_diff(phi_anal, phi)
plt.imshow(np.abs(np.transpose(diff.difference)), origin="lower",
           interpolation="nearest",
           extent=[g.xlim[0], g.xlim[1],
                   g.zlim[0], g.zlim[1]])
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("comparitive_difference10.png")
'''
# test convergence
'''
# analytical potential of a cube for the whole space
anal_phi = g.scratch_array()
for i in range(g.nx):
    for j in range(g.ny):
        for k in range(g.nz):
            cube_V = ap.Analytical_Cube_Potential(sites, g.x[i], g.y[j], g.z[k])
            anal_phi[i, j, k] = cube_V.V

lmax = 10
center = (a/2, b/2, c/2)
# center = (0.0, 0.0, 0.0)
m = multipole.Multipole(g, dens, lmax, 0.3*g.dr, center=center)
phi = m.Phi()
'''

# analytical potential of a spheroid for the whole space
anal_phi = g.scratch_array()
for i in range(g.nx):
    for j in range(g.ny):
        for k in range(g.nz):
            spheroid_V = ap.Ana_Mac_pot(a1, a3, g.x[i], g.y[j], g.z[k], density)
            anal_phi[i, j, k] = spheroid_V.potential

'''
# analytical potential of a sphere for the whole space
anal_phi = g.scratch_array()
for i in range(g.nx):
    for j in range(g.ny):
        for k in range(g.nz):
            sph_phi = ap.Ana_Sph_pot(g.x[i], g.y[j], g.z[k], a, density)
            anal_phi[i, j, k] = sph_phi.potential
            '''
# normalized L2 norm error
L2normerr = L2.L2_diff(anal_phi, phi, g)
print("resolution = ", nx)
print("lmax =", lmax)
print("L2 norm error is ", L2normerr)
