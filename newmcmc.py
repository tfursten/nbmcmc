from math import *
import numpy as np
import sympy.mpmath as sy
import scipy.misc as fac
import scipy.special as sp
import scipy.spatial.distance as scd
import pymc
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def sph_law_of_cos(u, v):
    '''Returns distance between two geographic
    coordinates in meters using spherical law of cosine'''
    R = 6371000
    u = np.radians(u)
    v = np.radians(v)
    delta_lon = v[1] - u[1]
    d = acos(sin(u[0]) * sin(v[0]) +
             cos(u[0]) * cos(v[0]) * cos(delta_lon)) * R
    return d


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
                 data_file, path="./", cartesian=True):
        self.mu = mu
        self.data_file = data_file
        self.path = path
        self.mu2 = -2.0 * self.mu
        self.z = exp(self.mu2)
        self.sqrz = sqrt(1.0 - self.z)
        self.g0 = log(1 / float(self.sqrz))
        self.nb_start = nb_start
        self.d_start = density_start
        self.plog = np.array([sy.polylog(i + 1, self.z)
                              for i in xrange(34)])

        self.n_markers = None
        self.marker_names = None
        self.dist = None
        self.sz = None
        self.ibd = None
        self.tsz = None
        self.fbar = None
        self.weight = None

        self.nb_prior_mu = None
        self.nb_prior_tau = None
        self.d_prior_mu = None
        self.d_prior_tau = None

    def set_prior_params(self, n_mu, n_tau, d_mu, d_tau):
        self.nb_prior_mu = n_mu
        self.nb_prior_tau = n_tau
        self.d_prior_mu = d_mu
        self.d_prior_tau = d_tau

    def parse_data(data_file, path, cartesian):
        data = np.array(np.genfromtxt(path + data_file,
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
        self.weight = 2 / (self.tsz - 1.0)

    def t_series(self, x, sigma):
        sum = 0.0
        pow2 = 1
        for t in xrange(34):
            dt = 2 * t
            pow2 <<= 1
            powX = 1.0
            powS = 1.0
            for i in xrange(dt):
                powX *= x
                powS *= sigma
            s = (self.plog[t] * powX) /\
                (fac.factorial2(dt, exact=True) * pow2 * powS)
            if((t % 2) == 0):
                sum += s
            else:
                sum -= s
        return sum

    def bessel(self, x, sigma):
        t = (x / float(sigma)) * self.sqrz
        return sp.k0(t)

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

        # deterministic function to calculate pIBD from Wright Malecot formula
        @pymc.deterministic(plot=False)
        def Phi():
            pIBD = np.repeat(self.fbar,
                             self.ibd.shape[1]).reshape(self.ibd.shape)
            negative_values = np.less(pIBD, 0)
            # Change any negative values to zero
            if np.any(negative_values):
                idx = np.where(negative_values == 1)
                pIBD[idx] = 2 ** (-52)
                # print("WARNING: pIBD fell below zero"
                # "for distance classes:", idx)
            return pIBD

        # Marginal Likelihoods
        Li = np.empty((self.nreps, self.ndc), dtype=object)
        Lsim = np.empty((self.nreps, self.ndc), dtype=object)
        Li = pymc.Container(
            [[pymc.Binomial('Li_{}_{}'.format(i, j), n=self.sz[i][j],
                            p=Phi[i][j], observed=True,
                            value=self.data[i][j])
              for j in xrange(self.ndc)] for i in xrange(self.nreps)])

        Lsim = pymc.Container([[pymc.Binomial('Lsim_{}_{}'.format(i, j),
                                              n=self.sz[i][j],
                                              p=Phi[i][j]) for j
                                in xrange(self.ndc)]
                               for i in xrange(self.nreps)])

        return locals()

    def make_model(self, data=None):
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
            phi = np.zeros((self.ibd.shape))
            phi_bar = 0
            denom = 4.0 * pi * nb + self.g0
            use_bessel = np.greater(self.dist, 5 * s)
            for index in np.ndindex(phi.shape):
                if use_bessel[index[1]]:
                    p = self.bessel(self.dist[index[1]], s) / denom
                else:
                    p = self.t_series(self.dist[index[1]], s) / denom
                phi_bar += p * self.sz[index[0]][index[1]]
                phi[index[0]][index[1]] = p
            phi_bar = np.divide(phi_bar, self.tsz)
            r = (phi.T - phi_bar) / (1.0 - phi_bar)
            pIBD = (self.fbar + (1-self.fbar) * r).T
            negative_values = np.less(pIBD, 0)
            # Change any negative values to zero
            if np.any(negative_values):
                idx = np.where(negative_values == 1)
                pIBD[idx] = 2 ** (-52)
                # print("WARNING: pIBD fell below zero"
                # "for distance classes:", idx)
            return pIBD

        # Marginal Likelihoods
        Li = np.empty((self.nreps, self.ndc), dtype=object)
        Lsim = np.empty((self.nreps, self.ndc), dtype=object)
        Li = pymc.Container(
            [[pymc.Binomial('Li_{}_{}'.format(i, j), n=self.sz[i][j],
                            p=Phi[i][j], observed=True,
                            value=self.data[i][j])
              for j in xrange(self.ndc)] for i in xrange(self.nreps)])

        Lsim = pymc.Container([[pymc.Binomial('Lsim_{}_{}'.format(i, j),
                                              n=self.sz[i][j],
                                              p=Phi[i][j]) for j
                                in xrange(self.ndc)]
                               for i in xrange(self.nreps)])

        return locals()

    def run_model(self, it, burn, thin, outfile, plot, model_com=False):
        dbname = outfile + ".pickle"
        M = pymc.Model(self.make_model())
        S = pymc.MCMC(
            M, db='pickle', calc_deviance=True,
            dbname=dbname)
        S.sample(iter=it, burn=burn, thin=thin)
        S.db.close()
        # for i in xrange(self.nreps):
        # for j in xrange(self.ndc):
        # S.Lsim[i][j].summary()
        # if plot:
        # pymc.Matplot.gof_plot(
        # S.Lsim[i][j], self.data[i][j],
        # name="gof" + str(i) + str(j))
        S.sigma.summary()
        S.ss.summary()
        S.density.summary()
        S.nb.summary()
        S.neigh.summary()
        reps = np.array([['Lsim_{}_{}'.format(i, j) for j in xrange(
            self.ndc)] for i in xrange(self.nreps)])
        S.write_csv(
            outfile + ".csv", variables=["sigma", "ss", "density",
                                         "nb", "neigh"] + list(reps.flatten()))
        S.stats()
        if plot:
            pymc.Matplot.plot(S.ss, format="pdf")
            pymc.Matplot.plot(S.neigh, format="pdf")
            pymc.Matplot.plot(S.density, format="pdf")
            pymc.Matplot.plot(S.sigma, format="pdf")
            pymc.Matplot.plot(S.nb, format="pdf")
            # [S.ss, S.neigh, S.density, S.sigma, S.nb,
            # S.lognb, S.logss, S.logs])
        # trace = S.trace("neigh")[:]
        if model_com:
            NM = pymc.Model(self.make_null_model())
            NS = pymc.MCMC(NM, db='pickle', calc_deviance=True,
                           dbname=outfile + "_null.pickle")
            NS.sample(iter=it, burn=burn, thin=thin)
            reps = np.array([['Lsim_{}_{}'.format(i, j) for j in xrange(
                self.ndc)] for i in xrange(self.nreps)])
            NS.write_csv(outfile + "_null.csv",
                         variables=["sigma", "ss", "density", "nb", "neigh"] +
                         list(reps.flatten()))
            # pymc.raftery_lewis(trace, q=0.025, r=0.01)
            hoDIC = NS.dic
            haDIC = S.dic
            com_out = open(outfile + "_model_comp.txt", 'w')
            com_out.write("Null Hypothesis DIC: " + str(hoDIC) + "\n")
            com_out.write("Alt Hypothesis DIC: " + str(haDIC) + "\n")
            NS.db.close()
        # pymc.gelman_rubin(S)
        S.db.close()
