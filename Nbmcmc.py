from math import *
import numpy as np
import numpy.ma as ma
import sympy.mpmath as sy
import scipy.misc as fac
import scipy.special as sp
import scipy.spatial.distance as scd
import pymc
import matplotlib.pyplot as plt
import gc
import time
from memory_profiler import profile
plt.style.use('ggplot')


def sph_law_of_cos(u, v):
    '''Returns distance between two geographic
    coordinates in meters using spherical law of cosine'''
    R = 6371000
    u = np.radians(u)
    v = np.radians(v)
    delta_lon = v[1] - u[1]
    return acos(sin(u[0]) * sin(v[0]) +
                cos(u[0]) * cos(v[0]) * cos(delta_lon)) * R


def ibd_count(m1, m2):
    count = 0
    for i in m1:
        if i == 0:
            continue
        for j in m2:
            if j == 0:
                continue
            if i == j:
                count += 1
    return count


def tot_count(m1, m2):
    total = 0
    for i in m1:
        if i == 0:
            continue
        for j in m2:
            if j == 0:
                continue
            total += 1
    return total


def tile_reshape(v, n, m):
    return np.tile(v, n).reshape(n, m)

# def equirect(u, v):
#    '''Returns distance between two geographic coordinates
#    in meters using equirectangular approximation'''
#    R = 6371000
#    u = np.radians(u)
#    v = np.radians(v)
#    delta_lon = v[1] - u[1]
#    delta_lat = v[0] - u[0]
#    x = delta_lon * cos((u[0] + v[0]) / 2.0)
#    y = delta_lat
#    d = sqrt(x * x + y * y) * R
#    return d


class NbMC:

    def __init__(self, mu, nb_start, density_start,
                 data_file, out_file, out_path="./", cartesian=True):
        self.mu = mu
        self.data_file = data_file
        self.path = path
        self.mu2 = -2.0 * self.mu
        self.z = exp(self.mu2)
        self.sqrz = sqrt(1.0 - self.z)
        self.g0 = log(1 / float(self.sqrz))
        self.nb_start = nb_start
        self.d_start = density_start
        self.taylor_terms = None
        self.t2 = None
        self.n_markers = None
        self.marker_names = None
        self.dist = None
        self.sz = None
        self.ibd = None
        self.tsz = None
        self.fbar = None
        self.fbar_1 = None
        self.weight = None

        self.parse_data(data_file, path, cartesian)
        self.set_taylor_terms()
        self.nb_prior_mu = None
        self.nb_prior_tau = None
        self.d_prior_mu = None
        self.d_prior_tau = None
        self.out_file = out_file
        self.out_path = out_path
        self.M = None
        self.S = None

    def set_prior_params(self, n_mu, n_tau, d_mu, d_tau):
        self.nb_prior_mu = n_mu
        self.nb_prior_tau = n_tau
        self.d_prior_mu = d_mu
        self.d_prior_tau = d_tau

    def parse_data(self, data_file, cartesian):
        data = np.array(np.genfromtxt(data_file,
                                      delimiter=",",
                                      dtype=str,
                                      skip_header=False,
                                      comments="#"))
        self.marker_names = data[0][3:]
        self.n_markers = len(self.marker_names)
        data = data[1:][:].T
        coords = np.array(data[:][1:3].T, dtype=float)
        if cartesian:
            self.dist = scd.pdist(coords, 'euclidean')
        else:
            self.dist = scd.pdist(coords, sph_law_of_cos)
        markers = np.array(data[:][3:], ndmin=2)
        markers = np.array(
            np.core.defchararray.split(markers, sep="/").tolist(), dtype=str)
        markers = np.core.defchararray.lower(markers)
        # order matters "na" needs to be after "nan"
        for n in ["none", "nan", "na", "x", "-", "."]:
            markers = np.core.defchararray.replace(markers, n, "0")
        markers = markers.astype(int, copy=False)
        self.ibd = np.array([scd.pdist(i, ibd_count) for i in markers],
                            dtype=int)
        self.sz = np.array([scd.pdist(i, tot_count) for i in markers],
                           dtype=int)
        self.tsz = np.sum(self.sz, axis=1, dtype=float)
        self.fbar = np.divide(np.sum(self.ibd, axis=1), self.tsz)
        self.fbar_1 = 1 - self.fbar
        self.weight = 2 / (self.tsz - 1.0)
        print len(self.dist)

    def set_taylor_terms(self):
        terms = 34
        n = len(self.dist)
        t = np.array([i for i in xrange(terms)])
        Li = tile_reshape(np.array([sy.polylog(i + 1, self.z)
                                    for i in xrange(terms)]), n, terms)
        fac2 = tile_reshape(fac.factorial(2 * t), n, terms)
        two2t = tile_reshape(2**(t + 1), n, terms)
        sign = tile_reshape((-1)**t, n, terms)
        self.t2 = tile_reshape(2 * t, n, terms)
        dist = np.repeat(self.dist, terms).reshape(n, terms)
        x2t = np.power(dist, self.t2)
        self.taylor_terms = np.divide(np.multiply(np.multiply(Li, x2t), sign),
                                      np.multiply(fac2, two2t))

    def t_series(self, mask, sigma):
        return ma.array(np.sum(
                        np.multiply(
                            np.divide(1, np.power(float(sigma),
                                                  self.t2)),
                            self.taylor_terms),
                        axis=1), mask=mask)

    def bessel(self, x, sigma):
        return sp.k0((x / float(sigma)) * self.sqrz)

    def make_null_model(self, data=None):
        nb = pymc.Lognormal('nb', value=self.nb_start,
                            mu=self.nb_prior_mu,
                            tau=self.nb_prior_tau)
        density = pymc.Lognormal('density', value=self.d_start,
                                 mu=self.d_prior_mu,
                                 tau=self.d_prior_tau)

        @pymc.deterministic
        def sigma(nb=nb, d=density):
            return sqrt(nb / float(d))

        @pymc.deterministic
        def ss(s=sigma):
            return s * s

        @pymc.deterministic
        def neigh(nb=nb):
            return 4.0 * nb * pi

        @pymc.deterministic(plot=False)
        def Phi():
            return ma.masked_less(np.repeat(self.fbar,
                                            self.ibd.shape[1]).reshape(
                                                self.ibd.shape),
                                  0).filled(0)

        @pymc.stochastic(observed=True)
        def marginal_bin(value=self.ibd, p=Phi, n=self.sz):
            return np.sum((value * np.log(p) + (n - value) *
                           np.log(1 - p)).T * self.weight)

        return locals()

    def make_model(self):
        nb = pymc.Lognormal('nb', value=self.nb_start,
                            mu=self.nb_prior_mu,
                            tau=self.nb_prior_tau)
        density = pymc.Lognormal('density', value=self.d_start,
                                 mu=self.d_prior_mu,
                                 tau=self.d_prior_tau)

        @pymc.deterministic
        def sigma(nb=nb, d=density):
            return sqrt(nb / float(d))

        @pymc.deterministic
        def ss(s=sigma):
            return s * s

        @pymc.deterministic
        def neigh(nb=nb):
            return 4.0 * nb * pi

        # deterministic function to calculate pIBD from Wright Malecot formula
        @pymc.deterministic(plot=False, trace=False)
        def Phi(nb=nb, s=sigma):
            denom = 4.0 * pi * nb + self.g0
            use_bessel = self.bessel(ma.masked_less_equal(self.dist, 5 * s,
                                                          copy=True), s)
            use_taylor = self.t_series(use_bessel.mask, s)
            phi = np.divide(use_bessel.filled(use_taylor), denom)
            phi_bar = np.divide(np.sum(np.multiply(self.sz, phi), axis=1),
                                self.tsz)
            return np.array((ma.masked_less((self.fbar + self.fbar_1 *
                                             ((tile_reshape(phi,
                                                            self.n_markers,
                                                            len(self.dist)).T -
                                               phi_bar) / (1 - phi_bar))),
                                            0)).filled(0), dtype=float).T

        @pymc.stochastic(observed=True)
        def marginal_bin(value=self.ibd, p=Phi):
            return np.sum((value * np.log(p) + (self.sz - value) *
                           np.log(1 - p)).T * self.weight)
        return locals()

    def run_model(self, it, burn, thin, plot=False):
        dbname = self.out_path + self.out_file + ".pickle"
        self.M = pymc.Model(self.make_model())
        self.S = pymc.MCMC(
            self.M, db='pickle', calc_deviance=True,
            dbname=dbname)
        self.S.sample(iter=it, burn=burn, thin=thin)
        self.S.ss.summary()
        self.S.sigma.summary()
        self.S.density.summary()
        self.S.nb.summary()
        self.S.neigh.summary()
        self.S.write_csv(self.out_path + self.out_file + ".csv",
                         variables=["sigma", "ss", "density", "nb", "neigh"])
        #self.S.stats()
        if plot:
            pymc.Matplot.plot(self.S, format="pdf", path=self.out_path,
                              suffix="_" + self.out_file)
        self.S.db.close()

    def model_comp(self, it, burn, thin):
        NM = pymc.Model(self.make_null_model())
        NS = pymc.MCMC(NM, db='pickle', calc_deviance=True,
                       dbname=self.out_path + self.out_file + "_null.pickle")
        NS.sample(iter=it, burn=burn, thin=thin)
        NS.write_csv(self.out_path + self.out_file + "_null.csv",
                         variables=["sigma", "ss", "density", "nb", "neigh"])
        ha = pymc.MAP(self.M)
        ho = pymc.MAP(NM)
        ha.fit()
        ho.fit()
        haBIC = ha.BIC
        hoBIC = ho.BIC
        haAIC = ha.AIC
        hoAIC = ho.AIC
        print hoAIC, hoBIC
        print haAIC, haBIC
        com_out = open(self.out_path + self.out_file + "_model_comp.txt", 'w')
        com_out.write("Null Hypothesis AIC: " + str(hoAIC) + "\n")
        com_out.write("Alt Hypothesis AIC: " + str(haAIC) + "\n")
        com_out.write("Null Hypothesis BIC: " + str(hoBIC) + "\n")
        com_out.write("Alt Hypothesis BIC: " + str(haBIC) + "\n")
        NS.db.close()
