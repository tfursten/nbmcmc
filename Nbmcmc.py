from math import *
import numpy as np
import shutil
import sympy.mpmath as sy
import scipy.misc as fac
import scipy.special as sp
import pymc
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class NbMC:

    def __init__(self, mu, ploidy, sigma_start, density_start,
                 data_in, dist_in, size_in, n_terms, data_is_raw=False):
        self.k = ploidy
        self.mu = mu
        self.mu2 = -2.0 * self.mu
        self.z = exp(self.mu2)
        self.sqrz = sqrt(1.0 - self.z)
        self.g0 = log(1 / float(self.sqrz))
        self.s_start = sigma_start
        self.d_start = density_start

        self.ndc = 0
        self.tsz = 0
        self.fbar = 0
        self.dist = np.empty(0, dtype=float)
        self.data = np.empty(0, dtype=int)
        self.sz = np.empty(0, dtype=int)
        self.set_data(data_in, dist_in, size_in, data_is_raw)
        self.plog = np.array([sy.polylog(i + 1, self.z)
                              for i in xrange(n_terms)])
        self.n_terms = n_terms
        self.s_prior_mu = sigma_start
        self.s_prior_tau = 1
        self.d_prior_mu = density_start
        self.d_prior_tau = 1

    def set_prior_params(self, s_mu, s_tau, d_mu, d_tau):
        self.s_prior_mu = s_mu
        self.s_prior_tau = s_tau
        self.d_prior_mu = d_mu
        self.d_prior_tau = d_tau

    def raw_to_dc(self, rawData):
        n = len(rawData)
        ibd = np.zeros(n / 2)
        sz = np.zeros(n / 2)
        for i in xrange(n):
            if np.isnan(rawData[i]):
                continue
            for j in xrange(i + 1, n):
                if np.isnan(rawData[j]):
                    continue
                k = abs(j - i)
                if k > (n / 2):
                    k = n - k
                if int(rawData[i]) == int(rawData[j]):
                    ibd[k - 1] += 1
                sz[k - 1] += 1
        return ibd, sz

    def sort_data(self, data, dc, sz):
        z = zip(dc, data, sz)
        z.sort()
        self.data = np.array([j for i, j, k in z], dtype=int)
        self.dist = np.array([i for i, j, k in z], dtype=float)
        self.sz = np.array([k for i, j, k in z], dtype=int)

    def set_data(self, newData, dc, sz, data_is_raw):
        # TODO: add more functionality for different types of data
        if data_is_raw:
            self.data, self.sz = raw_to_dc(self, newData)
        if type(sz) is int:
            sz = [sz for i in xrange(len(newData))]
        if len(newData) == len(dc) and len(dc) == len(sz):
            self.sort_data(newData, dc, sz)
            self.dist2 = self.dist ** 2
            self.tsz = np.sum(self.sz)
            self.ndc = len(self.dist)
            self.fbar = np.sum(self.data) / float(self.tsz)
        else:
            raise Exception(
                "ERROR: data and distance class arrays are not equal length")

    def t_series(self, x, sigma):
        sum = 0.0
        pow2 = 1
        for t in xrange(self.n_terms):
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

    def make_model(self, data=None):
        # sigma = pymc.Lognormal('sigma', mu=exp(self.s_prior_mu),
        #                       tau=self.s_prior_tau, value=exp(self.s_start))
        # density = pymc.Lognormal('density', mu=exp(self.d_prior_mu),
        #                         tau=self.d_prior_tau,
        #                         value=exp(self.d_start))
        sigma = pymc.TruncatedNormal('sigma', value=self.s_start,
                                     mu=self.s_prior_mu,
                                     tau=self.s_prior_tau,
                                     a=2 ** (-52),
                                     b=np.inf)
        density = pymc.TruncatedNormal('density', value=self.d_start,
                                       mu=self.d_prior_mu,
                                       tau=self.d_prior_tau,
                                       a=2 ** (-52),
                                       b=np.inf)

        #@pymc.deterministic
        # def ed(d=density):
        #    return log(d)

        #@pymc.deterministic
        # def es(s=sigma):
        #    return log(s)

        @pymc.deterministic
        def enb(s=sigma, d=density):
            return 2 * pi * s * s * d

        # deterministic function to calculate pIBD from Wright Malecot formula
        @pymc.deterministic(plot=False)
        def Phi(s=sigma, d=density):
            phi = np.zeros((self.ndc))
            phi_bar = 0
            denom = 2.0 * self.k * pi * s * s * d + self.g0
            split = self.ndc
            for i in xrange(self.ndc):
                if self.dist[i] > 6 * s:
                    split = i
                    break
            for sss in xrange(split):
                if self.dist[sss] == 0:
                    p = self.g0 / denom
                else:
                    p = self.t_series(self.dist[sss], s) / denom
                phi_bar += p * self.sz[sss]
                phi[sss] = p
            for lll in xrange(split, self.ndc):
                p = self.bessel(self.dist[lll], s) / denom
                phi_bar += p * self.sz[lll]
                phi[lll] = p
            phi_bar /= float(self.tsz)
            r = (phi - phi_bar) / (1.0 - phi_bar)
            pIBD = self.fbar + (1.0 - self.fbar) * r
            negative_values = np.less(pIBD, 0)
            # Change any negative values to zero
            if np.any(negative_values):
                idx = np.where(negative_values == 1)
                pIBD[idx] = 2 ** (-52)
                # print("WARNING: pIBD fell below zero"
                #"for distance classes:", idx)
            return pIBD

        # Marginal Likelihoods
        Li = np.empty(self.ndc, dtype=object)
        Lsim = np.empty(self.ndc, dtype=object)
        if data is None:
            Li = pymc.Container(
                [pymc.Binomial('Li_%d' % i, n=self.sz[i],
                               p=Phi[i], observed=True,
                               value=self.data[i]) for i in xrange(self.ndc)])
        else:
            Li = pymc.Container(
                [pymc.Binomial('Li_%i' % i, n=self.sz[i],
                               p=Phi[i], observed=True,
                               value=data) for i in xrange(self.ndc)])

        Lsim = pymc.Container([pymc.Binomial('Lsim_%i' % i,
                                             n=self.sz[i],
                                             p=Phi[i]) for i
                               in xrange(self.ndc)])

        return locals()

    def run_model(self, it, burn, thin, outfile, plot, nAdj=10):
        dbname = outfile + ".pickle"
        M = pymc.Model(self.make_model())
        S = pymc.MCMC(
            M, db='pickle', calc_deviance=False,
            dbname=dbname)
        S.sample(iter=it, burn=burn, thin=thin)
        S.trace('sigma')[:]
        S.trace('density')[:]
        S.trace('enb')[:]
        for i in xrange(self.ndc):
            S.trace('Lsim_%i' % i)[:]
            S.Lsim[i].summary()
        S.sigma.summary()
        S.density.summary()
        S.enb.summary()
        reps = ['Lsim_%i' % i for i in xrange(self.ndc)]
        S.write_csv(outfile, variables=["sigma", "density", "enb"] + reps)
        S.stats()
        if plot:
            pymc.Matplot.plot(S)
        S.db.close()
