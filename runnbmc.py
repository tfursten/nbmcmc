#!/usr/bin/env python
import argparse
import numpy as np
from Nbmcmc import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')

parser = argparse.ArgumentParser(
    description="Bayesian Estimation of Neighborhood Size "
    "using composite marginal likelihood")
parser.add_argument(
    "infile", metavar='INFILE', type=str,
    help="name of data file")
parser.add_argument(
    "outfile", metavar='OUTFILE', type=str,
    help="name of output file")
parser.add_argument(
    "-u", "--mu", default=0.0001, type=float,
    help="mutation rate")
parser.add_argument(
    "-k", "--ploidy", default=1.0, type=float,
    help="ploidy")
parser.add_argument(
    "-nb", "--nb_start", default=1.0, type=float,
    help="starting value for Neighborhood Size")
parser.add_argument(
    "-d", "--density_start", default=1.0,
    type=float, help="starting value for density")
parser.add_argument(
    "-t", "--n_terms", default=30, type=int,
    help="number of terms for taylor series")
parser.add_argument(
    "-it", "--iter", default=10000, type=int,
    help="number of MCMC iterations")
parser.add_argument(
    "-b", "--burn", default=100, type=int,
    help="length of burnin period for MCMC")
parser.add_argument(
    "-th", "--thin", default=1, type=int,
    help="thin the MCMC chains")
parser.add_argument(
    "-p", "--plot", action="store_true",
    help="output plots")
parser.add_argument(
    "-m", "--n_markers", required=True, type=int,
    help="number of markers")
parser.add_argument(
    "-n", "--n_ind", default=100, type=int,
    help="number of pairs per distance class")
parser.add_argument(
    "--nb_mu", default=1.0, type=float,
    help="mean for truncated normal neighborhood size prior")
parser.add_argument(
    "--nb_tau", default=0.01, type=float,
    help="precision for truncated normal neighborhood size prior")
parser.add_argument(
    "--d_mu", default=1.0, type=float,
    help="mean for density truncated normal prior")
parser.add_argument(
    "--d_tau", default=0.01, type=float,
    help="precision for density truncated normal prior")
parser.add_argument(
    "-l", "--line", default=0, type=int,
    help="line of file with data")
args = parser.parse_args()

# TODO parse different types of data files
mcmctot = args.iter / args.thin
s = str("Outfile: {}\nInfile: {}\nMu: {}\nPloidy: {}\n"
        "Nb Start: {}\nDensity Start: {}\n"
        "Nb Mu: {}\nNb Tau: {}\nDensity Mu: {}\n"
        "Density Tau: {}\n"
        "Taylor Series Terms: {}\nMCMC iterations: {}\n"
        "MCMC burn: {}\nMCMC thin: {}\n"
        "MCMC total: {}").format(args.outfile, args.infile, args.mu,
                                 args.ploidy, args.nb_start,
                                 args.density_start, args.nb_mu, args.nb_tau,
                                 args.d_mu, args.d_tau, args.n_terms,
                                 args.iter, args.burn, args.thin, mcmctot)
print(s)
param = open("parameters.txt", 'w')
param.write(s)
param.close()


data = np.array(np.genfromtxt(args.infile, delimiter=",", dtype=int))
#data = data[args.line]
ndc = len(data)
dist = np.array([i + 1 for i in xrange(ndc)])
sz = [args.n_ind * args.n_markers for i in xrange(ndc - 1)]
sz.append(args.n_ind / 2 * args.n_markers)
sz = np.array(sz)

nbmc = NbMC(args.mu, args.ploidy, args.nb_start,
            args.density_start, data, dist, sz,
            args.n_terms)
nbmc.set_prior_params(args.nb_mu, args.nb_tau, args.d_mu, args.d_tau)
nbmc.run_model(
    args.iter, args.burn, args.thin, args.outfile + str(args.line), args.plot)
sz = np.array(sz, dtype=float)
data = np.divide(data, sz)

f = open(args.outfile + str(args.line) + ".csv", 'r')
dist = [i + 1 for i in xrange(50)]
for i in xrange(6):
    f.readline()
upper = []
lower = []
means = []
for line in f:
    line = line.strip().split(",")
    means.append(float(line[1]))
    lower.append(float(line[1]) - float(line[6]))
    upper.append(float(line[10]) - float(line[1]))
means = np.divide(np.array(means), sz)
upper = np.divide(np.array(upper), sz)
lower = np.divide(np.array(lower), sz)
plt.clf()
plt.errorbar(dist, means, yerr=[lower, upper])
plt.plot(dist, data, 'o')
plt.ylim(0, 1)
plt.xlim(0, 51)
plt.savefig(args.outfile + str(args.line) + ".png")
# plt.show()
