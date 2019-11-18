from numba import jit
import numpy as np
import scipy.constants as sc


@jit
def calcLegPolyL(l, x):
    # Calculate the Legendre polynomials. We use a stable recurrence relation:
    # (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x).
    # This uses initial conditions:
    # P_0(x) = 1
    # P_1(x) = x
    if (l == 0):
        return 1.0
    elif(l == 1):
        return x
    else:
        legPolyL2 = 1.0
        legPolyL1 = x
        legPolyL = 0.0
        for n in range(2, l+1):
            legPolyL = ((2*n - 1) * x * legPolyL1 - (n-1) * legPolyL2) / n
            legPolyL2 = legPolyL1
            legPolyL1 = legPolyL
        return legPolyL


class Multipole():
    '''The Multipole Expansion in 2d case'''

    def __init__(self, grid, density, lmax, dr, center=(0.0, 0.0, 0.0)):

        self.dr = dr
        self.dr_mp = dr

        self.center = center
        self.g = grid
        self.lmax = lmax

        self.center = center
        self.x = self.g.x3d - self.center[0]
        self.y = self.g.y3d - self.center[1]
        self.z = self.g.z3d - self.center[2]
        self.radius = np.sqrt(self.x**2+self.y**2+self.z**2)
        self.cosTheta = self.z/self.radius
        self.m = density * self.g.vol

        # compute the bins, or the radius of the concentric sphere, r_mu
        x_max = max(abs(self.g.xlim[0] - center[0]), abs(self.g.xlim[1] -
                                                         center[0]))
        y_max = max(abs(self.g.ylim[0] - center[0]), abs(self.g.ylim[1] -
                                                         center[1]))
        z_max = max(abs(self.g.zlim[0] - center[1]), abs(self.g.zlim[1] -
                                                         center[2]))

        dmax = np.sqrt(x_max**2 + y_max**2 + z_max**2)

        self.n_bins = int(dmax/dr)

        # bin boundaries
        self.r_bin = np.linspace(0.0, dmax, self.n_bins)

    @jit
    def calcSolHarm(self, l):
        # calculate the solid harmonic function R_lm and I_lm in
        # eq 15 and eq 16

        P_l = self.g.scratch_array()
        self.R_l = self.g.scratch_array()
        self.I_l = self.g.scratch_array()

        for i in range(self.g.nx):
            for j in range(self.g.ny):
                for k in range(self.g.nz):
                    P_l[i, j, k] = calcLegPolyL(l, self.cosTheta[i, j, k])

        self.R_l = self.radius**l*P_l
        self.I_l = self.radius**(-l-1)*P_l

    @jit
    def calcML(self):
        # calculate the outer and inner multipole moment function
        # M_lm^R and M_lm^I in eq 17 and eq 18

        self.m_r = np.zeros((self.n_bins), dtype=np.float64)
        self.m_i = np.zeros((self.n_bins), dtype=np.float64)

        for i in range(self.n_bins):
            imask = self.radius <= self.r_bin[i]
            omask = self.radius > self.r_bin[i]
            self.m_r[i] += np.sum(self.R_l[imask] * self.m[imask])
            self.m_i[i] += np.sum(self.I_l[omask] * self.m[omask])

    @jit
    def sample_mtilde(self, r):
        # calculate the interpolated multipole moment M_lm^R^tilde
        # and M_lm^I^tilde in eq 19

        # we need to find which be we are in
        mu_m = np.argwhere(self.r_bin <= r)[-1][0]
        mu_p = np.argwhere(self.r_bin > r)[0][0]

        assert mu_p == mu_m + 1

        mtilde_r = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                           ) * self.m_r[mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_r[mu_m]

        mtilde_i = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                           ) * self.m_i[mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_i[mu_m]
        return mtilde_r, mtilde_i

    @jit
    def calcMulFaceY(self, dx, dy, dz, j, l):
        # calculate the contribution of M_lm^R^tilde * conj(I_lm) +
        # conj(M_lm^I^tilde)*R_lm
        # at the face of the cell
        # for the plane perpendicular to y
        # j is the index of the y coordinate

        # x, y, z coordinates of all surfaces of grid cell
        x = self.x + dx
        y = self.y + dy
        z = self.z + dz

        radius = np.sqrt(x**2 + y**2 + z**2)

        cosTheta = z/radius

        mulFace_l = self.g.scratch_y_plane_array()
        mtilde_r = self.g.scratch_y_plane_array()
        mtilde_i = self.g.scratch_y_plane_array()
        R_l = self.g.scratch_y_plane_array()
        I_l = self.g.scratch_y_plane_array()

        for i in range(self.g.nx):
            for k in range(self.g.nz):
                mtilde_r[i, k], mtilde_i[i, k] = self.sample_mtilde(radius[i, j, k])
                P_l = calcLegPolyL(l, cosTheta[i, j, k])

                R_l[i, k] = radius[i, j, k]**l * P_l
                I_l[i, k] = radius[i, j, k]**(-l-1) * P_l

        mulFace_l = mtilde_r * I_l + mtilde_i * R_l

        return mulFace_l

    @jit
    def PhiY(self, j):
        # calculate the potential for the plane perpendicular to y
        # j is the index of the y coordinate
        dx = self.g.dx/2
        dy = self.g.dy/2
        dz = self.g.dz/2

        '''
        area_x = self.g.dy*self.g.dz
        area_y = self.g.dz*self.g.dz
        area_z = self.g.dx*self.g.dy
        total_area = 2*(area_x+area_y+area_z)
        '''

        phiY = self.g.scratch_y_plane_array()

        # for mass distribution symmetric with respect to center of expansion
        for l in range(0, self.lmax+1, 2):
            self.calcSolHarm(l)
            self.calcML()
            MulFace_minus_x = self.calcMulFaceY(-dx, 0, 0, j, l)
            MulFace_minus_y = self.calcMulFaceY(0, -dy, 0, j, l)
            MulFace_minus_z = self.calcMulFaceY(0, 0, -dz, j, l)
            MulFace_plus_x = self.calcMulFaceY(dx, 0, 0, j, l)
            MulFace_plus_y = self.calcMulFaceY(0, dy, 0, j, l)
            MulFace_plus_z = self.calcMulFaceY(0, 0, dz, j, l)

            phiY += (MulFace_minus_x + MulFace_minus_y +
                     MulFace_minus_z + MulFace_plus_x +
                     MulFace_plus_y + MulFace_plus_z)/6

        # return phiY
        return -sc.G*phiY

    @jit
    def calcMulFace(self, dx, dy, dz, l):
        # calculate the contribution of M_lm^R^tilde * conj(I_lm) +
        # conj(M_lm^I^tilde)*R_lm at the surface
        # evaluated at the face of the cell

        # should 3d
        # rho and z coordinates of all surfaces of grid cell
        # x, y, z coordinates of all surfaces of grid cell
        x = self.x + dx
        y = self.y + dy
        z = self.z + dz

        radius = np.sqrt(x**2 + y**2 + z**2)

        cosTheta = z/radius

        mulFace_l = self.g.scratch_array()
        mtilde_r = self.g.scratch_array()
        mtilde_i = self.g.scratch_array()
        R_l = self.g.scratch_array()
        I_l = self.g.scratch_array()

        for i in range(self.g.nx):
            for j in range(self.g.ny):
                for k in range(self.g.nz):
                    mtilde_r[i, j, k], mtilde_i[i, j, k] = self.sample_mtilde(radius[i, j, k])

                    P_l = calcLegPolyL(l, cosTheta[i, j, k])

                    R_l[i, j, k] = radius[i, j, k]**l * P_l
                    I_l[i, j, k] = radius[i, j, k]**(-l-1) * P_l

        mulFace_l = mtilde_r * I_l + mtilde_i * R_l

        return mulFace_l

    @ jit
    def Phi(self):
        phi = self.g.scratch_array()
        '''
        dx = self.g.dx/2
        dy = self.g.dy/2
        dz = self.g.dz/2
        for l in range(0, self.lmax+1, 2):
            # in case where the mass is symetric, the multipole of odd l vanish
            self.calcSolHarm(l)
            self.calcML()

            MulFace_minus_x = self.calcMulFace(-dx, 0, 0, l)
            MulFace_minus_y = self.calcMulFace(0, -dy, 0, l)
            MulFace_minus_z = self.calcMulFace(0, 0, -dz, l)
            MulFace_plus_x = self.calcMulFace(dx, 0, 0, l)
            MulFace_plus_y = self.calcMulFace(0, dy, 0, l)
            MulFace_plus_z = self.calcMulFace(0, 0, dz, l)

            phi += (MulFace_minus_x + MulFace_minus_y +
                    MulFace_minus_z + MulFace_plus_x +
                    MulFace_plus_y + MulFace_plus_z)/6
                    '''

        # test with cell center
        phi = self.g.scratch_array()

        for l in range(0, self.lmax+1, 2):
            # in case where the mass is symetric, the multipole of odd l vanish
            self.calcSolHarm(l)
            self.calcML()
            MulFace = self.calcMulFace(0, 0, 0, l)
            phi += MulFace

        phi = -sc.G*phi

        return phi


"""
@jit
def calcR_lm(l, m, x, y, z):
    # to calculate the solid harmonic functions
    # use the recurrence relation in Flash User Guide:
    # R_{0,0}^c=1
    # R_{l,l}^c=-(xR_{l-1,l-1}^c-yR_{l-1,l-1}^s)/(2l)
    # R_{l,l}^s=-(yR_{l-1,l-1}^c+xR_{l-1,l-1}^s)/(2l)
    # R_{l,m}^{c/s}=((2l-1)zR_{l-1,m}^{c/s}-r^2R_{l-2,m}^{c/s})/((l+m)(l-m))
    # use R_{m, m} and R_{m+1, m}^{c,s}=zR_{m,m}^{c/s}
    # as initial conditions
    # compute the R_{m,m}^c and the R_{m,m}^s
    R_mmc = 1
    R_mms = 0
    if(l == 0):
        return R_mmc, R_mms
    else:
        R_m1m1c = R_mmc
        R_m1m1s = R_mms
        for n in range(1, m+1):
            R_mmc = -(x*R_m1m1c-y*R_m1m1s)/(2*n)
            R_mms = -(y*R_m1m1c+x*R_m1m1s)/(2*n)
            R_m1m1c = R_mmc
            R_m1m1s = R_mms
        if(l == m):
            return R_mmc, R_mms
        else:
            # compute the R_{m+1,m}^c and R_{m+1,m}^s
            R_m1mc = z*R_mmc
            R_m1ms = z*R_mms
            if(l == m+1):
                return R_m1mc, R_m1ms
            else:
                # compute the general R_{l,m}^c and the R_{l,m}^s
                # use R_{m, m} and R_{m+1, m}^{c,s}=zR_{m,m}^{c/s}
                R_l1mc = R_m1mc
                R_l2mc = R_mmc
                R_l1ms = R_m1ms
                R_l2ms = R_mms
                r = np.sqrt(x**2+y**2+z**2)
                for n in range(m+2, l+1):
                    R_lmc = ((2*n-1)*z*R_l1mc-r**2*R_l2mc)/((n+m)*(n-m))
                    R_lms = ((2*n-1)*z*R_l1ms-r**2*R_l2ms)/((n+m)*(n-m))
                    R_l2mc = R_l1mc
                    R_l1mc = R_lmc
                    R_l2ms = R_l1ms
                    R_l1ms = R_lms
                return R_lmc, R_lms


@jit
def calcI_lm(l, m, x, y, z):
    # to calculate the solid harmonic functions
    # use the recurrence relation in Flash User Guide:
    # I_{0,0}^c=1/r
    # I_{l,l}^c=-(2l-1)(xI_{l-1,l-1}^c-yI_{l-1,l-1}^s)/(r^2)
    # R_{l,l}^s=-(2l-1)(yI_{l-1,l-1}^c+xI_{l-1,l-1}^s)/(r^2)
    # I_{l,m}^{c/s}=((2l-1)zI_{l-1,m}^{c/s}-((l-1)^2-m^2)I_{l-2,m}^{c/s})/(r^2)
    # use I_{m, m} and I_{m+1, m}^{c,s}=(2m+1)zI_{m,m}^{c/s}/(r^2)
    # as initial conditions
    r = np.sqrt(x**2+y**2+z**2)
    I_mmc = 1/r
    I_mms = 0
    if(l == 0):
        return I_mmc, I_mms
    else:
        I_m1m1c = I_mmc
        I_m1m1s = I_mms
        for n in range(1, m+1):
            I_mmc = -(2*n-1)*(x*I_m1m1c-y*I_m1m1s)/(r**2)
            I_mms = -(2*n-1)*(y*I_m1m1c+x*I_m1m1s)/(r**2)
            I_m1m1c = I_mmc
            I_m1m1s = I_mms
        if(l == m):
            return I_mmc, I_mms
        else:
            # compute the R_{m+1,m}^c and R_{m+1,m}^s
            I_m1mc = z*I_mmc
            I_m1ms = z*I_mms
            if(l == m+1):
                return I_m1mc, I_m1ms
            else:
                # compute the general R_{l,m}^c and the R_{l,m}^s
                # use R_{m, m} and R_{m+1, m}^{c,s}=zR_{m,m}^{c/s}
                I_l1mc = I_m1mc
                I_l2mc = I_mmc
                I_l1ms = I_m1ms
                I_l2ms = I_mms
                for n in range(m+2, l+1):
                    I_lmc = ((2*n-1)*z*I_l1mc-((n-1)**2-m**2)*I_l2mc)/r**2
                    I_lms = ((2*n-1)*z*I_l1ms-((n-1)**2-m**2)*I_l2ms)/r**2
                    I_l2mc = I_l1mc
                    I_l1mc = I_lmc
                    I_l2ms = I_l1ms
                    I_l1ms = I_lms
                return I_lmc, I_lms


class Multipole():
    def __init__(self, grid, density, lmax, dr, center=(0.0, 0.0, 0.0)):
        '''The Multipole Expansion in 2d case'''

        self.g = grid
        self.lmax = lmax
        self.dr_mp = dr
        self.center = center
        self.x = self.g.x3d - self.center[0]
        self.y = self.g.y3d - self.center[1]
        self.z = self.g.z3d - self.center[2]
        self.radius = np.sqrt(self.x**2+self.y**2+self.z**2)
        self.m = density * self.g.vol

        # compute the bins, or the radius of the concentric sphere, r_mu
        x_max = max(abs(self.g.xlim[0] - center[0]), abs(self.g.xlim[1] -
                                                         center[0]))
        y_max = max(abs(self.g.ylim[0] - center[0]), abs(self.g.ylim[1] -
                                                         center[1]))
        z_max = max(abs(self.g.zlim[0] - center[1]), abs(self.g.zlim[1] -
                                                         center[2]))

        dmax = np.sqrt(x_max**2 + y_max**2 + z_max**2)

        self.n_bins = int(dmax/dr)

        # bin boundaries
        self.r_bin = np.linspace(0.0, dmax, self.n_bins)

    @jit
    def calcSolHarm(self, l, m):
        # this calculate the m_r and m_i indexed l, m,
        # defined in eq. 17 and eq. 18
        # density is density that lives on a grid

        # loop over cells
        self.Rlmc = self.g.scratch_array()
        self.Rlms = self.g.scratch_array()
        self.Ilmc = self.g.scratch_array()
        self.Ilms = self.g.scratch_array()
        for i in range(self.g.nx):
            for j in range(self.g.ny):
                for k in range(self.g.nz):
                    self.Rlmc[i, j, k], self.Rlms[i, j, k] = calcR_lm(l, m,
                                                                      self.x[i, j, k],
                                                                      self.y[i, j, k],
                                                                      self.z[i, j, k])
                    self.Ilmc[i, j, k], self.Ilms[i, j, k] = calcI_lm(l, m,
                                                                      self.x[i, j, k],
                                                                      self.y[i, j, k],
                                                                      self.z[i, j, k])

    @jit
    def calcML(self):
        # calculate the outer and inner multipole moment function
        # M_lm^R and M_lm^I in eq 17 and eq 18

        self.m_rc = np.zeros((self.n_bins), dtype=np.float64)
        self.m_rs = np.zeros((self.n_bins), dtype=np.float64)
        self.m_ic = np.zeros((self.n_bins), dtype=np.float64)
        self.m_is = np.zeros((self.n_bins), dtype=np.float64)

        for i in range(self.n_bins):
            imask = self.radius <= self.r_bin[i]
            omask = self.radius > self.r_bin[i]
            self.m_rc[i] += np.sum(self.Rlmc[imask] * self.m[imask])
            self.m_rs[i] += np.sum(self.Rlms[imask] * self.m[imask])
            self.m_ic[i] += np.sum(self.Ilmc[omask] * self.m[omask])
            self.m_is[i] += np.sum(self.Ilms[omask] * self.m[omask])

    @jit
    def sample_mtilde(self, r):
        # this returns the result of Eq. 19
        # r is the radius of the point of the field from the expansion center
        # we need to find out the index of r_mu_plus and r_mu_minus in Eq. 19
        mu_m = np.argwhere(self.r_bin <= r)[-1][0]
        mu_p = np.argwhere(self.r_bin > r)[0][0]

        assert mu_p == mu_m + 1

        mtilde_rc = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                            ) * self.m_rc[mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_rc[mu_m]

        mtilde_rs = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                            ) * self.m_rs[mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_rs[mu_m]

        mtilde_ic = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                            ) * self.m_ic[mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_ic[mu_m]

        mtilde_is = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                            ) * self.m_is[mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_is[mu_m]

        return mtilde_rc, mtilde_rs, mtilde_ic, mtilde_is

    @jit
    def calcMulFaceY(self, dx, dy, dz, j, l, m):
        # calculate the contribution of M_lm^R^tilde * conj(I_lm) +
        # conj(M_lm^I^tilde)*R_lm
        # at the face of the cell
        # for the plane perpendicular to y
        # j is the index of the y coordinate

        # x, y, z coordinates of all surfaces of grid cell
        x = self.x + dx
        y = self.y + dy
        z = self.z + dz

        radius = np.sqrt(x**2 + y**2 + z**2)

        mulFace_lm = self.g.scratch_y_plane_array()
        mtilde_rc = self.g.scratch_y_plane_array()
        mtilde_rs = self.g.scratch_y_plane_array()
        mtilde_ic = self.g.scratch_y_plane_array()
        mtilde_is = self.g.scratch_y_plane_array()
        Rlmc = self.g.scratch_y_plane_array()
        Rlms = self.g.scratch_y_plane_array()
        Ilmc = self.g.scratch_y_plane_array()
        Ilms = self.g.scratch_y_plane_array()

        for i in range(self.g.nx):
            for k in range(self.g.nz):
                mtilde_rc[i, k], mtilde_rs[i, k], mtilde_ic[i, k], mtilde_is[i, k] = self.sample_mtilde(
                    radius[i, j, k])

                Rlmc[i, k], Rlms[i, k] = calcR_lm(l, m, x[i, j, k],
                                                  y[i, j, k], z[i, j, k])
                Ilmc[i, k], Ilms[i, k] = calcI_lm(l, m, x[i, j, k],
                                                  y[i, j, k], z[i, j, k])

        if(m == 0):
            mulFace_lm = mtilde_rc*Ilmc+mtilde_ic*Rlmc
        else:
            mulFace_lm = 2*(mtilde_rc*Ilmc+mtilde_rs*Ilms +
                            mtilde_ic*Rlmc+mtilde_is*Rlms)

        return mulFace_lm

    @ jit
    def PhiY(self, j):
        # calculate the potential for the plane perpendicular to y
        # j is the index of the y coordinate
        dx = self.g.dx/2
        dy = self.g.dy/2
        dz = self.g.dz/2

        '''
        area_x = self.g.dy*self.g.dz
        area_y = self.g.dz*self.g.dz
        area_z = self.g.dx*self.g.dy
        total_area = 2*(area_x+area_y+area_z)
        '''

        phiY = self.g.scratch_y_plane_array()

        # for mass distribution symmetric with respect to center of expansion
        for l in range(0, self.lmax+1, 2):
            for m in range(0, l+1):
                self.calcSolHarm(l, m)
                self.calcML()
                MulFace_minus_x = self.calcMulFaceY(-dx, 0, 0, j, l, m)
                MulFace_minus_y = self.calcMulFaceY(0, -dy, 0, j, l, m)
                MulFace_minus_z = self.calcMulFaceY(0, 0, -dz, j, l, m)
                MulFace_plus_x = self.calcMulFaceY(dx, 0, 0, j, l, m)
                MulFace_plus_y = self.calcMulFaceY(0, dy, 0, j, l, m)
                MulFace_plus_z = self.calcMulFaceY(0, 0, dz, j, l, m)
                '''
                phiY += -sc.G*(MulFace_minus_x + MulFace_minus_y +
                               MulFace_minus_z + MulFace_plus_x +
                               MulFace_plus_y + MulFace_plus_z)/6
                               '''
                phiY += (MulFace_minus_x + MulFace_minus_y +
                         MulFace_minus_z + MulFace_plus_x +
                         MulFace_plus_y + MulFace_plus_z)/6

        # return phiY
        return -sc.G*phiY

    @jit
    def calcMulFace(self, dx, dy, dz, l, m):
        # calculate the contribution of M_lm^R^tilde * conj(I_lm) +
        # conj(M_lm^I^tilde)*R_lm
        # at the face of the cell
        # for all the points in the space

        # x, y, z coordinates of all surfaces of grid cell
        x = self.x + dx
        y = self.y + dy
        z = self.z + dz

        radius = np.sqrt(x**2 + y**2 + z**2)

        mulFace_lm = self.g.scratch_array()
        mtilde_rc = self.g.scratch_array()
        mtilde_rs = self.g.scratch_array()
        mtilde_ic = self.g.scratch_array()
        mtilde_is = self.g.scratch_array()
        Rlmc = self.g.scratch_array()
        Rlms = self.g.scratch_array()
        Ilmc = self.g.scratch_array()
        Ilms = self.g.scratch_array()

        for i in range(self.g.nx):
            for j in range(self.g.ny):
                for k in range(self.g.nz):
                    mtilde_rc[i, j, k], mtilde_rs[i, j, k], mtilde_ic[i, j, k], mtilde_is[i, j, k] = self.sample_mtilde(
                        radius[i, j, k])

                    Rlmc[i, j, k], Rlms[i, j, k] = calcR_lm(l, m, x[i, j, k],
                                                            y[i, j, k], z[i, j, k])
                    Ilmc[i, j, k], Ilms[i, j, k] = calcI_lm(l, m, x[i, j, k],
                                                            y[i, j, k], z[i, j, k])

        if(m == 0):
            mulFace_lm = mtilde_rc*Ilmc+mtilde_ic*Rlmc
        else:
            mulFace_lm = 2*(mtilde_rc*Ilmc+mtilde_rs*Ilms +
                            mtilde_ic*Rlmc+mtilde_is*Rlms)

        return mulFace_lm

    @ jit
    def Phi(self):
        # calculate the potential for all the points in the space
        dx = self.g.dx/2
        dy = self.g.dy/2
        dz = self.g.dz/2

        '''
        area_x = self.g.dy*self.g.dz
        area_y = self.g.dz*self.g.dz
        area_z = self.g.dx*self.g.dy
        total_area = 2*(area_x+area_y+area_z)
        '''

        phi = self.g.scratch_array()

        '''
        # for mass distribution symmetric with respect to center of expansion
        for l in range(0, self.lmax+1, 2):
            for m in range(0, l+1):
                self.calcSolHarm(l, m)
                self.calcML()
                MulFace_minus_x = self.calcMulFace(-dx, 0, 0, l, m)
                MulFace_minus_y = self.calcMulFace(0, -dy, 0, l, m)
                MulFace_minus_z = self.calcMulFace(0, 0, -dz, l, m)
                MulFace_plus_x = self.calcMulFace(dx, 0, 0, l, m)
                MulFace_plus_y = self.calcMulFace(0, dy, 0, l, m)
                MulFace_plus_z = self.calcMulFace(0, 0, dz, l, m)

                phi += (MulFace_minus_x + MulFace_minus_y +
                        MulFace_minus_z + MulFace_plus_x +
                        MulFace_plus_y + MulFace_plus_z)/6
                        '''
        for l in range(0, self.lmax+1, 2):
            # for the axysymmetric case
            self.calcSolHarm(l, 0)
            self.calcML()
            MulFace_minus_x = self.calcMulFace(-dx, 0, 0, l, 0)
            MulFace_minus_y = self.calcMulFace(0, -dy, 0, l, 0)
            MulFace_minus_z = self.calcMulFace(0, 0, -dz, l, 0)
            MulFace_plus_x = self.calcMulFace(dx, 0, 0, l, 0)
            MulFace_plus_y = self.calcMulFace(0, dy, 0, l, 0)
            MulFace_plus_z = self.calcMulFace(0, 0, dz, l, 0)

            phi += (MulFace_minus_x + MulFace_minus_y +
                    MulFace_minus_z + MulFace_plus_x +
                    MulFace_plus_y + MulFace_plus_z)/6
        # return phi
        return -sc.G*phi
        """
