from math import *
import numpy as np
import sympy.mpmath as sy
import scipy.misc as fac
import scipy.special as sp
import pymc
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class NbMC:

    def __init__(self, mu, ploidy, nb_start, density_start,
                 data_in, dist_in, size_in, n_terms, n_samples, data_is_raw=False):
        self.mu = mu
        self.k = ploidy
        self.mu2 = -2.0 * self.mu
        self.z = exp(self.mu2)
        self.sqrz = sqrt(1.0 - self.z)
        self.g0 = log(1 / float(self.sqrz))
        self.nb_start = nb_start
        self.d_start = density_start
        self.nreps = 0
        self.ndc = 0
        self.tsz = 0
        self.fbar = 0
        self.const = 2/(n_samples-1.0)
        self.dist = np.empty(0, dtype=float)
        self.data = np.empty(0, dtype=int)
        self.sz = np.empty(0, dtype=int)
        self.set_data(data_in, dist_in, size_in)
        self.plog = np.array([sy.polylog(i + 1, self.z)
                              for i in xrange(n_terms)])
        self.n_terms = n_terms
        self.nb_prior_mu = nb_start
        self.nb_prior_tau = 0.001
        self.d_prior_mu = density_start
        self.d_prior_tau = 0.001

    def set_prior_params(self, n_mu, n_tau, d_mu, d_tau):
        self.nb_prior_mu = n_mu
        self.nb_prior_tau = n_tau
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



    def set_data(self, newData, dc, sz):
        # if data_is_raw:
            #self.data, self.sz = raw_to_dc(self, newData)
        if len(newData[0]) == len(dc[0]) and len(dc[0]) == len(sz[0]):
            self.data = newData
            self.dist = dc
            self.sz = sz
            self.dist2 = self.dist ** 2
            self.tsz = np.sum(self.sz[0])
            self.ndc = len(self.dist[0])
            self.fbar = np.array(
                [np.sum(d) / float(self.tsz) for d in self.data])
            self.nreps = len(self.data)
            print self.nreps
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
            phi = np.zeros((self.ndc))
            r = phi
            pIBD = np.array(
                [self.fbar[i] + (1.0 - self.fbar[i]) * r
                 for i in xrange(self.nreps)])
            negative_values = np.less(pIBD, 0)
            # Change any negative values to zero
            if np.any(negative_values):
                idx = np.where(negative_values == 1)
                pIBD[idx] = 2 ** (-52)
                # print("WARNING: pIBD fell below zero"
                #"for distance classes:", idx)
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
            phi = np.zeros((self.ndc))
            phi_bar = 0
            denom = 4.0 * pi * nb + self.g0
            split = self.ndc
            for i in xrange(self.ndc):
                if self.dist[0][i] > 5 * s:
                    split = i
                    break
            for sss in xrange(split):
                if self.dist[0][sss] == 0:
                    p = self.g0 / denom
                else:
                    p = self.t_series(self.dist[0][sss], s) / denom
                phi_bar += p * self.sz[0][sss]
                phi[sss] = p
            for lll in xrange(split, self.ndc):
                p = self.bessel(self.dist[0][lll], s) / denom
                phi_bar += p * self.sz[0][lll]
                phi[lll] = p
            phi_bar /= float(self.tsz)
            r = (phi - phi_bar) / (1.0 - phi_bar)

            pIBD = np.array(
                [self.fbar[i] + (1.0 - self.fbar[i]) * r
                 for i in xrange(self.nreps)])
            negative_values = np.less(pIBD, 0)
            # Change any negative values to zero
            if np.any(negative_values):
                idx = np.where(negative_values == 1)
                pIBD[idx] = 2 ** (-52)
                # print("WARNING: pIBD fell below zero"
                #"for distance classes:", idx)
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
        #S.Lsim[i][j], self.data[i][j],
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
                                                "nb", "neigh"]
            + list(reps.flatten()))
        S.stats()
        if plot:
            pymc.Matplot.plot(S.ss, format="pdf")
            pymc.Matplot.plot(S.neigh, format="pdf")
            pymc.Matplot.plot(S.density, format="pdf")
            pymc.Matplot.plot(S.sigma, format="pdf")
            pymc.Matplot.plot(S.nb, format="pdf")
            #[S.ss, S.neigh, S.density, S.sigma, S.nb,
            # S.lognb, S.logss, S.logs])
        #trace = S.trace("neigh")[:]
        if model_com:
            NM = pymc.Model(self.make_null_model())
            NS = pymc.MCMC(NM, db='pickle', calc_deviance=True,
                           dbname=outfile + "_null.pickle")
            NS.sample(iter=it, burn=burn, thin=thin)
            reps = np.array([['Lsim_{}_{}'.format(i, j) for j in xrange(
                self.ndc)] for i in xrange(self.nreps)])
            NS.write_csv(outfile + "_null.csv", variables=["sigma", "ss", "density",
                                                              "nb", "neigh"]
                         + list(reps.flatten()))
            #pymc.raftery_lewis(trace, q=0.025, r=0.01)
            hoDIC = NS.dic
            haDIC = S.dic
            com_out = open(outfile + "_model_comp.txt", 'w')
            com_out.write("Null Hypothesis DIC: " + str(hoDIC) + "\n")
            com_out.write("Alt Hypothesis DIC: " + str(haDIC) + "\n")
            NS.db.close()
        # pymc.gelman_rubin(S)
        S.db.close()
