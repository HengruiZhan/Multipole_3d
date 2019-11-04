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


nx = 32
ny = 32
nz = 32

xlim = (-1.0, 1.0)
ylim = (-1.0, 1.0)
zlim = (-1.0, 1.0)
g = grid.Grid(nx, ny, nz, xlim, ylim, zlim)
dens = g.scratch_array()
"""
# density of the sphere
sph_center = (0.0, 0.0, 0.3)
radius = np.sqrt((g.x3d - sph_center[0])**2 + (g.y3d - sph_center[1])**2 +
                 (g.z3d - sph_center[2])**2)
dens[radius <= 0.3] = 1.0
"""

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
        cube_V = ap.Analytical_Cube_Potential(sites, g.x[i], g.y[5], g.z[j])
        phi_anal[i, j] = cube_V.V

lmax = 0
center = (a/2, b/2, c/2)
# center = (0.0, 0.0, 0.0)
m = multipole.Multipole(g, dens, lmax, 0.3*g.dr, center=center)

phi = g.scratch_y_plane_array()

phi = m.PhiY(5)

'''
plt.imshow(np.log10(np.abs(np.transpose(phi))), origin="lower",
           interpolation="nearest",
           extent=[g.xlim[0], g.xlim[1],
                   g.zlim[0], g.zlim[1]])

plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("lmax=10.png")
'''
'''
# test convergence
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

# normalized L2 norm error
L2normerr = L2.L2_diff(anal_phi, phi, g)

print("lmax =", lmax)
print(L2normerr)
'''
'''
diff = comdf.comp_diff(phi_anal, phi)
plt.imshow(np.abs(np.transpose(diff.difference)), origin="lower",
           interpolation="nearest",
           extent=[g.xlim[0], g.xlim[1],
                   g.zlim[0], g.zlim[1]])
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("comparitive_difference.png")
'''
