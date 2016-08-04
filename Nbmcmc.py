from math import *
import numpy as np
import numpy.ma as ma
import mpmath as sy
import scipy.misc as fac
import scipy.special as sp
import scipy.spatial.distance as scd
import scipy.stats as ss
import pymc
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('ggplot')

blue = (0/255.0, 142/255.0, 214/255.0)
grey = (79/255.0, 85/255.0, 87/255.0)
lite_grey = (175/255.0, 165/255.0, 147/255.0)


def log_normal_pdf(x, mu, sigma):
    return 1/(sqrt(2 * pi) * sigma * x) * exp((-(log(x)-mu)**2)/(2*sigma**2))
log_norm_vec = np.vectorize(log_normal_pdf)


def sph_law_of_cos(u, v):
    '''Returns distance between two geographic
    coordinates in meters using spherical law of cosine'''
    R = 6371000
    u = np.radians(u)
    v = np.radians(v)
    delta_lon = v[1] - u[1]
    return acos(sin(u[0]) * sin(v[0]) +
                cos(u[0]) * cos(v[0]) * cos(delta_lon)) * R

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
                 data_file, out_file, out_path="./", sep="\t",
                 cartesian=True):
        self.mu = mu
        self.ploidy = None
        self.data_file = data_file
        self.out_file = out_file
        self.out_path = out_path
        self.mu2 = -2.0 * self.mu
        self.z = exp(self.mu2)
        self.sqrz = sqrt(1.0 - self.z)
        self.g0 = log(1 / float(self.sqrz))
        self.nb_start = nb_start
        self.d_start = density_start
        self.taylor_terms = None
        self.t2 = None
        self.n_markers = None
        self.n_pairs = None
        self.marker_names = None
        self.dist = None
        self.unique_dists = None
        self.unique_ID = None
        self.unique_counts = None
        self.n_dist_class = None
        self.iis = None
        self.n = None
        self.fbar = None
        self.fbar_1 = None
        self.weights = None
        self.n_alleles = None
        self.n_ind = None
        self.parse_data(data_file, cartesian, sep)
        self.set_taylor_terms()
        self.nb_prior_mu = None
        self.nb_prior_tau = None
        self.d_prior_mu = None
        self.d_prior_tau = None
        self.M = None
        self.S = None

    def set_prior_params(self, n_mu, n_tau, d_mu, d_tau):
        self.nb_prior_mu = log(n_mu)
        self.nb_prior_tau = n_tau
        self.d_prior_mu = log(d_mu)
        self.d_prior_tau = d_tau

    def adjust_weight_for_null(self, marker_idx, ind_idx, allele_idx, weights):
        '''In the case of a null allele the weight (number of comparisons) for
        every other individual is reduced by 1 and the weight is Inf for the
        null allele (1/Inf = 0). The weight for all other alleles within this
        individual will remain the same.'''
        idx1 = int(ind_idx * self.ploidy)
        idx2 = int(idx1 + self.ploidy)
        weights[marker_idx][idx1:idx2] = np.add(
                                        weights[marker_idx][idx1:idx2], 1)
        weights[marker_idx][:] = np.subtract(weights[marker_idx][:], 1)
        weights[marker_idx][idx1 + allele_idx] = np.Inf
        return weights

    def set_weights(self, markers):
        weights = np.array(
                            [[self.n_alleles - self.ploidy for i in xrange(
                              self.n_alleles)]
                                for j in xrange(self.n_markers)], dtype=float)
        nulls = np.where(markers == 0)
        for m, i, k in zip(nulls[0], nulls[1], nulls[2]):
            weights = self.adjust_weight_for_null(m, i, k, weights)
        return np.divide(1., weights)

    def parse_data(self, data_file, cartesian, sep):
        data = np.array(np.genfromtxt(data_file,
                                      delimiter=sep,
                                      dtype=str,
                                      skip_header=False,
                                      comments="#"))
        self.marker_names = data[0][3:]
        data = data[1:][:].T
        coords = np.array(data[:][1:3].T, dtype=float)
        if cartesian:
            dist = scd.squareform(scd.pdist(coords, 'euclidean'))
        else:
            dist = scd.squareform(scd.pdist(coords, sph_law_of_cos))
        markers = np.core.defchararray.lower(
                                            np.array(
                                             np.core.defchararray.split(
                                              np.array(data[:][3:], ndmin=2),
                                              sep="/").tolist(), dtype=str))
        # order matters "na" needs to be after "nan"
        for n in ["none", "nan", "na", "x", "-", "."]:
            markers = np.core.defchararray.replace(markers, n, "0")
        markers = markers.astype(int, copy=False)
        self.n_markers, self.n_ind, self.ploidy = markers.shape
        self.n_alleles = self.n_ind * self.ploidy
        self.n_pairs = np.sum([i for i in xrange(self.n_alleles)]) - \
            (self.n_alleles / self.ploidy)
        iis = []
        pair_dist = []
        pair_list = []
        pair_weights = []
        weights = self.set_weights(markers)

        for m in xrange(self.n_markers):
            this_iis = []
            this_weight = []
            for i in xrange(self.n_ind-1):
                for k in xrange(self.ploidy):
                    for j in xrange(i+1, self.n_ind):
                        for l in xrange(self.ploidy):
                            if markers[m, i, k] == 0 or markers[m, j, l] == 0:
                                this_iis.append(np.nan)
                            elif markers[m, i, k] == markers[m, j, l]:
                                this_iis.append(1)
                            else:
                                this_iis.append(0)
                            this_weight.append(weights[m][i*self.ploidy+k] +
                                               weights[m][j*self.ploidy+l])
                            if m == 0:
                                pair_list.append([[m, i, k], [m, j, l]])
                                pair_dist.append(dist[i, j])
            iis.append(this_iis)
            pair_weights.append(this_weight)
        pair_weights = np.array(pair_weights)
        self.dist = np.array(pair_dist)
        self.pairs = np.array(pair_list)
        iis = np.array(iis, dtype=float)
        self.fbar = np.divide(np.nansum(iis, axis=1),
                              np.subtract(self.n_pairs,
                              np.sum(np.isnan(iis), axis=1)))
        self.fbar_1 = np.subtract(1, self.fbar)
        self.unique_dists, self.unique_ID, self.unique_counts = np.unique(
                                                           self.dist,
                                                           return_inverse=True,
                                                           return_counts=True)
        self.n_dist_class = len(self.unique_dists)
        self.iis = np.array([[np.nansum(iis[j][np.where(self.unique_ID == i)])
                            for i in xrange(self.n_dist_class)]
                            for j in xrange(self.n_markers)], dtype=float)
        self.n = np.array([[iis[j][np.where(self.unique_ID == i)].shape[0] -
                          np.sum(
                          np.isnan(iis[j][np.where(self.unique_ID == i)]))
                          for i in xrange(self.n_dist_class)]
                          for j in xrange(self.n_markers)], dtype=float)
        self.weights = np.array([[np.nansum(pair_weights[j][np.where(
                                self.unique_ID == i)])
                                for i in xrange(self.n_dist_class)]
                                for j in xrange(self.n_markers)],
                                dtype=float)

    def set_taylor_terms(self):
        terms = 34
        t = np.array([i for i in xrange(terms)])
        Li = np.array([sy.polylog(i + 1, self.z) for i in xrange(terms)])
        fac2 = fac.factorial2(2 * t)
        two2t = 2**(t + 1)
        sign = (-1)**t
        self.t2 = 2 * t
        dist = np.repeat(self.unique_dists, terms).reshape(self.n_dist_class,
                                                           terms)
        x2t = np.power(dist, self.t2)
        self.taylor_terms = np.divide(np.multiply(np.multiply(Li, x2t), sign),
                                      np.multiply(fac2, two2t))

    def t_series(self, mask, sigma):
        return ma.array(np.sum(
                        np.multiply(
                            np.divide(1., np.power(float(sigma), self.t2)),
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
        def phi():
            return ma.masked_less(
               np.tile(
                 self.fbar, (self.n_dist_class, 1)), 0).filled(
                                                                2 ** (-52)).T

        @pymc.stochastic(observed=True)
        def marginal_binomial(value=self.iis, p=phi):
            return np.sum((value * np.log(p) + (self.n - value) *
                           np.log(1 - p)) * self.weights)

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
            return sqrt(nb / d)

        @pymc.deterministic
        def ss(s=sigma):
            return s * s

        @pymc.deterministic
        def neigh(nb=nb):
            return 4.0 * nb * pi

        # deterministic function to calculate pIBD from Wright Malecot formula
        @pymc.deterministic(plot=False, trace=False)
        def phi(nb=nb, s=sigma):
            denom = 4.0 * pi * nb + self.g0
            use_bessel = self.bessel(ma.masked_less_equal(self.unique_dists,
                                                          5 * s,
                                                          copy=True), s)
            use_taylor = self.t_series(use_bessel.mask, s)
            phi = np.tile(np.divide(use_bessel.filled(use_taylor), denom),
                          (self.n_markers, 1))
            phi_bar = np.divide(np.sum(np.multiply(self.n, phi), axis=1),
                                np.sum(self.n, axis=1))
            p = np.divide(np.subtract(phi.T, phi_bar),
                          np.subtract(1, phi_bar)).T
            p = (self.fbar + self.fbar_1 * p.T).T
            p = np.array((ma.masked_less(p, 0)).filled(2 ** (-52)),
                         dtype=float)
            return p

        # cml = np.empty((self.n_markers, self.n_dist_class), dtype=object)
        # cml = pymc.Container([[pymc.Binomial('cml_{}_{}'.format(i, j),
        #                      n=self.n[i][j],
        #                      p=phi[i][j], observed=True,
        #                      value=self.iis[i][j])
        #        for i in xrange(self.n_markers)]
        #       for j in xrange(self.n_dist_class)])

        cml_rep = np.empty((self.n_markers, self.n_dist_class), dtype=object)
        cml_rep = pymc.Container([[pymc.Binomial('cml_rep_{}_{}'.format(i, j),
                                 n=self.n[i][j],
                                 p=phi[i][j])
                                 for i in xrange(self.n_markers)]
                                 for j in xrange(self.n_dist_class)])

        @pymc.stochastic(observed=True)
        def marginal_bin(value=self.iis, p=phi):
            return np.sum((value * np.log(p) + (self.n - value) *
                           np.log(1 - p)) * self.weights)

        return locals()

    def run_model(self, it, burn, thin, plot_diog=False, plot_ppc=False,
                  plot_prior=False):
        dbname = self.out_path + self.out_file + ".pickle"
        self.M = pymc.Model(self.make_model())
        self.S = pymc.MCMC(
            self.M, db='pickle', calc_deviance=True,
            dbname=dbname)
        self.S.sample(iter=it, burn=burn, thin=thin)
        self.S.neigh.summary()
        self.S.sigma.summary()
        self.S.density.summary()
        self.S.write_csv(self.out_path + self.out_file + ".csv",
                         variables=["sigma", "ss", "density", "nb", "neigh"])
        if plot_prior:
            self.plot_prior()
        if plot_ppc:
            self.posterior_check()
        if plot_diog:
            self.plot_diognostics()

        self.S.db.close()
        return len(self.S.neigh.trace())

    def plot_diognostics(self):
        pymc.Matplot.plot(self.S.neigh, format="pdf", path=self.out_path,
                          suffix="_" + self.out_file)
        pymc.Matplot.plot(self.S.sigma, format="pdf", path=self.out_path,
                          suffix="_" + self.out_file)
        pymc.Matplot.plot(self.S.density, format="pdf", path=self.out_path,
                          suffix="_" + self.out_file)

    def plot_prior(self):
        print "Plotting neighborhood size prior vs. marginal posterior\n"
        trace = self.S.neigh.trace()
        x = np.linspace(0, np.max(trace)*1.1, 10000)[1:]
        y = log_norm_vec(x, self.nb_prior_mu, 1/sqrt(self.nb_prior_tau))

        y_vals, x_vals, _ = plt.hist(trace, normed=True, color=grey)

        plt.axhline(y=0, ls="solid", color=lite_grey)
        plt.axvline(x=0, ls="solid", color=lite_grey)
        plt.plot(x, y, '-', lw=1, label='Log-Normal Prior', color=blue)
        plt.ylim(-0.1, max(max(y), max(y_vals)) * 1.1)
        plt.xlim(-0.1, max(x_vals) * 1.1)
        plt.savefig(self.out_path+self.out_file+"_prior.pdf",
                    bbox_inches='tight')
        plt.close()

    def posterior_check(self):
        print "Plotting Posterior Predictive Check\n"

        upper_quant = np.array([[self.S.stats()["cml_rep_{}_{}".format(
                    i, j)]["quantiles"][97.5]
                    for i in xrange(self.n_markers)]
                   for j in xrange(self.n_dist_class)]).T
        lower_quant = np.array([[self.S.stats()["cml_rep_{}_{}".format(
                    i, j)]["quantiles"][2.5]
                    for i in xrange(self.n_markers)]
                   for j in xrange(self.n_dist_class)]).T
        mean = np.array([[self.S.stats()["cml_rep_{}_{}".format(
                    i, j)]["mean"]
                   for i in xrange(self.n_markers)]
                  for j in xrange(self.n_dist_class)]).T

        dist = self.unique_dists
        fig, ax = plt.subplots(self.n_markers, 1, sharex=True,
                               figsize=(3, 1.5 * self.n_markers))

        fig.tight_layout()

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=None, hspace=0.3)
        mean = np.divide(mean, self.n)
        upper = np.subtract(np.divide(upper_quant, self.n), mean)
        lower = np.subtract(mean, np.divide(lower_quant, self.n))

        max_y = np.amax(upper + mean) * 1.1
        min_y = np.amin(mean-lower) * 0.90
        for i in xrange(self.n_markers):
            ax[i].axhline(y=self.fbar[i], color=lite_grey, ls='solid',
                          zorder=0)
            ax[i].errorbar(dist, mean[i], yerr=[lower[i],
                           upper[i]], color=blue, zorder=1)
            ax[i].plot(dist, np.divide(self.iis[i], self.n[i]), '.',
                       color=grey,
                       markeredgewidth=0.0, zorder=2)
            ax[i].set_ylim(min_y, max_y)
            ax[i].tick_params(axis="both")
            ax[i].set_ylabel(self.marker_names[i], fontsize=9)
        fig.text(0.5, 0.0, 'Distance (Meters)', ha='center', fontsize=9)
        plt.savefig(self.out_path+self.out_file+"_ppc.pdf",
                    bbox_inches='tight')

    def model_comp(self, it, burn, thin):
        print "Running Model Comparison\n"
        NM = pymc.Model(self.make_null_model())
        NS = pymc.MCMC(NM, db='pickle', calc_deviance=True,
                       dbname=self.out_path + self.out_file + "_null.pickle")
        NS.sample(iter=it, burn=burn, thin=thin)
        NS.write_csv(self.out_path + self.out_file + "_null.csv",
                     variables=["sigma", "ss", "density", "nb", "neigh"])
        ha = self.S.DIC
        ho = NS.DIC
        com_out = open(self.out_path + self.out_file + "_model_comp.txt", 'w')
        com_out.write("Null Hypothesis DIC: " + str(ho) + "\n")
        com_out.write("Alt Hypothesis DIC: " + str(ha) + "\n")
        com_out.write("Delta DIC: " + str(ho - ha) + "\n")
        com_out.write("Rel. Delta DIC: " + str((ho - ha)/ho) + "\n")
        NS.db.close()
