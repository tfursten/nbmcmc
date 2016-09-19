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
import random

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


def grid_dist(i, j, mx):
    dix, diy = i // mx, i % mx
    djx, djy = j // mx, j % mx
    return scd.euclidean([dix, diy], [djx, djy])

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
                 data_file, out_file, bins, independent=False,
                 weight=True, out_path="./", sep="\t",
                 cartesian=True):
        self.mu = mu
        self.ploidy = None
        self.data_file = data_file
        self.out_file = out_file
        self.out_path = out_path
        self.bins = np.array(bins, ndmin=1)
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
        self.dist_class = None
        self.dist_avg = None
        self.unique_dists = None
        self.n_dist_class = None
        self.iis = None
        self.n = None
        self.n_scaled = None
        self.weights = None
        self.fbar = None
        self.fbar_1 = None
        self.n_alleles = None
        self.n_ind = None
        if independent:
            self.parse_ind_data(data_file, cartesian, sep)
        else:
            self.parse_data(data_file, cartesian, sep, weight)
        self.set_taylor_terms()
        self.nb_prior_mu = None
        self.nb_prior_tau = None
        self.d_prior_mu = None
        self.d_prior_tau = None
        self.M = None
        self.S = None

    def set_prior_params(self, n_mu, n_tau, d_mu, d_tau):
        self.nb_prior_mu = n_mu
        self.nb_prior_tau = n_tau
        self.d_prior_mu = d_mu
        self.d_prior_tau = d_tau

    def get_independent_pairs(self, dists):
        # Pick independent pairs
        if self.bins.size == 1:
            self.bins = np.linspace(np.min(dists), np.max(dists),
                                    int(self.bins))
        d = np.digitize(dists, self.bins)
        uni, counts = np.unique(d, return_counts=True)
        freq = np.round(counts/float(np.sum(counts)) * self.n_ind//2)
        pairs = []

        for k in xrange(self.ploidy):
            these_pairs = []
            go = True
            finish = False
            count = 0
            ind = np.arange(self.n_ind).tolist()
            random.shuffle(ind)
            arr = [0 for i in xrange(len(freq))]
            while go:
                dd = grid_dist(ind[0], ind[1], sqrt(self.n_ind))
                b = int(np.digitize(dd, self.bins)) - 1
                if arr[b] >= freq[b] and not finish:
                    random.shuffle(ind)
                    count += 1
                else:
                    arr[b] += 1
                    these_pairs.append([ind.pop(0), ind.pop(0)])
                count += 1
                if len(ind) < 2:
                    go = False
                if count >= 1000:
                    finish = True
            pairs.append(these_pairs)
        return np.array(pairs).T

    def parse_ind_data(self, data_file, cartesian, sep):
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
        self.n_alleles = np.apply_over_axes(np.sum,
                                            np.array(markers,
                                                     dtype=bool),
                                            [1, 2]).flatten()
        iis = []
        pair_dist = []
        pair_list = []
        for m in xrange(self.n_markers):
            this_iis = []
            pairs = self.get_independent_pairs(scd.squareform(dist))
            for k, ploidy in enumerate(pairs):
                for pair in ploidy:
                    if markers[m, pair[0], k] == 0 or markers[m, pair[1], k] == 0:
                        this.iis.append(np.nan)
                    elif markers[m, pair[0], k] == markers[m, pair[1], k]:
                        this_iis.append(1)
                    else:
                        this_iis.append(0)
                    if m == 0:
                        pair_dist.append(dist[pair[0], pair[1]])
                        pair_list.append([[m, pair[0], k], [m, pair[1], k]])
            iis.append(this_iis)

        self.dist = np.array(pair_dist)
        self.pairs = np.array(pair_list)
        iis = np.array(iis, dtype=float)
        # set distance classes
        self.dist_class = np.digitize(self.dist, self.bins)
        self.unique_dists = np.unique(self.dist_class)
        self.n_dist_class = self.unique_dists.size
        self.dist_avg = np.array([np.mean(
                                          self.dist[np.where(
                                           self.dist_class == d)])
                                  for d in self.unique_dists])
        self.iis = np.array([[np.nansum(iis[j][np.where(self.dist_class == i)])
                            for i in self.unique_dists]
                            for j in xrange(self.n_markers)], dtype=float)
        self.n = np.array([[iis[j][np.where(self.dist_class == i)].shape[0] -
                          np.sum(
                          np.isnan(iis[j][np.where(self.dist_class == i)]))
                          for i in self.unique_dists]
                          for j in xrange(self.n_markers)], dtype=float)
        self.fbar = np.divide(np.nansum(self.iis, axis=1),
                              np.nansum(self.n, axis=1))
        self.fbar_1 = np.subtract(1, self.fbar)
        self.weights = 1
        self.n_scaled = self.n
        self.weights = (self.n_alleles//2)/np.nansum(self.n, axis=1)


    def parse_data(self, data_file, cartesian, sep, do_weight):
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
        self.n_alleles = np.apply_over_axes(np.sum,
                                            np.array(markers,
                                                     dtype=bool),
                                            [1, 2]).flatten()
        iis = []
        pair_dist = []
        pair_list = []

        for m in xrange(self.n_markers):
            this_iis = []
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
                            if m == 0:
                                pair_list.append([[m, i, k], [m, j, l]])
                                pair_dist.append(dist[i, j])
            iis.append(this_iis)
        self.dist = np.array(pair_dist)
        self.pairs = np.array(pair_list)
        iis = np.array(iis, dtype=float)
        # set distance classes
        # if bins is an integer evenly divide distances into n bins
        if self.bins.size == 1:
            self.bins = np.linspace(np.min(self.dist), np.max(self.dist),
                                    int(self.bins))
        self.dist_class = np.digitize(self.dist, self.bins)
        self.unique_dists = np.unique(self.dist_class)
        self.n_dist_class = self.unique_dists.size
        self.dist_avg = np.array([np.mean(
                                          self.dist[np.where(
                                           self.dist_class == d)])
                                  for d in self.unique_dists])
        self.iis = np.array([[np.nansum(iis[j][np.where(self.dist_class == i)])
                            for i in self.unique_dists]
                            for j in xrange(self.n_markers)], dtype=float)
        self.n = np.array([[iis[j][np.where(self.dist_class == i)].shape[0] -
                          np.sum(
                          np.isnan(iis[j][np.where(self.dist_class == i)]))
                          for i in self.unique_dists]
                          for j in xrange(self.n_markers)], dtype=float)
        self.fbar = np.divide(np.nansum(self.iis, axis=1),
                              np.nansum(self.n, axis=1))
        self.fbar_1 = np.subtract(1, self.fbar)
        if do_weight:
            self.weights = (self.n_alleles//2)/np.nansum(self.n, axis=1)
            # scaled total counts used when generating replicated data
            self.n_scaled = np.array((self.n.T * self.weights).T, dtype=int)
            self.n_scaled[np.where(self.n_scaled == 0)] = 1
        else:
            self.weights = 1
            self.n_scaled = self.n

    def get_distance_classes(self):
        d_map = {dc: davg for dc, davg in zip(self.unique_dists,
                                              self.dist_avg)}
        avg_dist = np.array([d_map[d] for d in self.dist_class])
        counts = np.array(np.mean(self.n, axis=0), dtype=str)
        scaled_counts = np.array(np.mean(self.n_scaled, axis=0),
                                 dtype=str)
        pairs = np.array([[p[0][1], p[1][1]] for p in self.pairs]).T
        d_info = np.stack((pairs[0], pairs[1], self.dist,
                           self.dist_class, avg_dist)).T
        d_info = d_info[::2]
        return {"bins": np.array(self.bins, dtype=str),
                "avg_dist": np.array(self.dist_avg, dtype=str),
                "dist_data": d_info,
                "counts": counts,
                "scaled_counts": scaled_counts}

    def set_taylor_terms(self):
        terms = 34
        t = np.array([i for i in xrange(terms)])
        Li = np.array([sy.polylog(i + 1, self.z) for i in xrange(terms)])
        fac2 = fac.factorial2(2 * t)
        two2t = 2**(t + 1)
        sign = (-1)**t
        self.t2 = 2 * t
        dist = np.repeat(self.dist_avg, terms).reshape(self.n_dist_class,
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
                           np.log(1 - p)).T * self.weights)

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
            use_bessel = self.bessel(ma.masked_less_equal(self.dist_avg,
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
                                 n=self.n_scaled[i][j],
                                 p=phi[i][j])
                                 for i in xrange(self.n_markers)]
                                 for j in xrange(self.n_dist_class)])

        @pymc.stochastic(observed=True)
        def marginal_bin(value=self.iis, p=phi):
            return np.sum((value * np.log(p) + (self.n - value) *
                           np.log(1 - p)).T * self.weights)

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

        y_vals, x_vals, _ = plt.hist(trace, normed=True, color=grey, bins=10)

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

        dist = self.dist_avg
        fig, ax = plt.subplots(self.n_markers, 1, sharex=True,
                               figsize=(3, 1.5 * self.n_markers))

        fig.tight_layout()

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=None, hspace=0.3)
        mean = np.divide(mean, self.n_scaled)
        upper = np.subtract(np.divide(upper_quant, self.n_scaled), mean)
        lower = np.subtract(mean, np.divide(lower_quant, self.n_scaled))

        max_y = np.amax(upper + mean) * 1.1
        min_y = np.amin(mean - lower)
        min_y = min_y * 0.90 if min_y > 0.1 else -0.1
        for i in xrange(self.n_markers):
            ax[i].axhline(y=self.fbar[i], color=lite_grey, ls='solid',
                          zorder=1)
            ax[i].errorbar(dist, mean[i], yerr=[lower[i],
                           upper[i]], color=blue, zorder=2)
            ax[i].plot(dist, np.divide(self.iis[i], self.n[i]), '.',
                       color=grey,
                       markeredgewidth=0.0, zorder=3)
            ax[i].set_ylim(min_y, max_y)
            ax[i].tick_params(axis="both")
            ax[i].set_ylabel(self.marker_names[i], fontsize=9)
        fig.text(0.5, 0.0, 'Distance (Meters)', ha='center', fontsize=9,
                 color=grey)
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
