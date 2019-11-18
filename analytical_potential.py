import numpy as np
import scipy.constants as sc


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


class Ana_Mac_pot(object):
    # use bisection method
    # Analytical potential of MacLaurin spheroid
    # to avoid the round off error, I reorganize the equations 21 to 27

    def __init__(self, a1, a3, x, y, z, density):
        assert a1 > a3
        e = np.sqrt((a1-a3)*(a1+a3))/a1

        # potential inside the sphere
        self.potential = 0.0

        if z**2/a3**2+(x**2+y**2)/a1**2 <= 1:
            # e^2
            e_sqr = (a1-a3)*(a1+a3)/a1**2
            A1 = np.sqrt(1 - e_sqr)/(e*e_sqr)*np.arcsin(e)-1/e_sqr+1
            A3 = 2/(e_sqr)-2*np.sqrt(1 - e_sqr)/e**3*np.arcsin(e)

            self.potential = -np.pi*sc.G*density *\
                (2*A1*a1**2-A1*(x**2+y**2) +
                 A3*(a3-z)*(a3+z))

        else:
            # b = a_1**2+a_3**2-rho**2-z**2
            # c = a_1**2*a_3**2-a_3**2 * rho**2 - z**2*a_1**2
            # lambda is the positive root of the eq. 27 of arXiv:1307.3135
            # lambda = (-b+sqrt(b**2-4ac))/2a
            # lam = (-b+np.sqrt(b**2-4*c))/2

            '''
            lam = (-a1**2-a3**2+rho**2+z**2)/2 +\
                np.sqrt(a1**4-2*a1**2*a3**2-2*a1**2*rho**2 +
                        2*a1**2*z**2+a3**4+2*a3**2*rho**2-2 *
                        a3**2*z**2+rho**4+2*rho**2*z**2+z**4)/2
                                          '''
            b = a1**2+a3**2-(x**2+y**2+z**2)
            c = a1**2*a3**2-a3**2*(x**2+y**2)-a1**2*z**2
            lam = (-b+np.sqrt(b**2-4*c))/2
            h = a1*e/np.sqrt(a3**2+lam)
            self.potential = -2*a3/(e**2)*np.pi*sc.G * density *\
                (a1*e*np.arctan(h)-1/2 * ((x**2+y**2) * (np.arctan(h) -
                                                         h/(1+h**2)) + 2*z**2 *
                                          (h-np.arctan(h)))/np.sqrt(a1**2-a3**2))


class Ana_Sph_pot(object):
    """Analytical potential of a perfect sphere"""

    def __init__(self, x, y, z, a, density):
        r = np.sqrt(x**2+y**2+z**2)

        self.potential = 0.0

        M = 4/3*np.pi * a**3 * density

        if r <= a:
            self.potential = -sc.G*M*(3*a**2-r**2)/(2*a**3)

        else:
            self.potential = -sc.G*M/r
