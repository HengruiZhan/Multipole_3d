import numpy as np


"""
class Analytical_Cube_Potential(object):
    '''Ana_Cube calculate the analytical potential the cube at one point.
    The sites have 3 elements (a, b ,c), and a, b, and c is the length
    of the sites x, y, z of the cube respectively. The potential is calculated
    at {The Newtonian Potential of a Homogeneous Cube} by Jorg Waldvogel.
    The rectangular are defined by 0<=x=<a, 0<=y<=b, 0<=z<=c'''

    def __init__(self, sites, x, y, z):
        self.sites = sites
        self.V = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    xi = self.x_i(i, x)
                    yj = self.y_i(j, y)
                    zk = self.z_i(k, z)
                    rijk = np.sqrt(xi**2+yj**2+zk**2)
                    # bug: for the first 3 terms, it is arctanh, not arctan
                    self.V += xi*yj*np.arctanh(zk/rijk) +\
                        yj*zk*np.arctanh(xi/rijk) +\
                        zk*xi*np.arctanh(yj/rijk) -\
                        xi**2/2*np.arctan((yj*zk)/(xi*rijk)) -\
                        yj**2/2*np.arctan((zk*xi)/(yj*rijk)) -\
                        zk**2/2*np.arctan((xi*yj)/(zk*rijk))

    def x_i(self, i, x):
        if i == 0:
            return x
        if i == 1:
            return self.sites[0] - x

    def y_i(self, j, y):
        if j == 0:
            return y
        if j == 1:
            return self.sites[1] - y

    def z_i(self, k, z):
        if k == 0:
            return z
        if k == 1:
            return self.sites[2] - z
            """

class Analytical_Cube_Potential(object):
