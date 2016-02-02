#!/usr/bin/env python
import argparse
import time
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
    "-t", "--n_terms", default=34, type=int,
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
    "-n", "--n_ind", default=100, type=int,
    help="number of pairs per distance class")
parser.add_argument(
    "--nb_mu", default=1.0, type=float,
    help="mean for truncated normal neighborhood size prior")
parser.add_argument(
    "--nb_tau", default=0.001, type=float,
    help="precision for truncated normal neighborhood size prior")
parser.add_argument(
    "--d_mu", default=1.0, type=float,
    help="mean for density truncated normal prior")
parser.add_argument(
    "--d_tau", default=0.001, type=float,
    help="precision for density truncated normal prior")
args = parser.parse_args()


start_time = time.time()

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


data = np.array(np.genfromtxt(args.infile, delimiter=",", dtype=int))
if len(data) == 1:
    data = np.array(data)
# data = data[args.line]
ndc = len(data[0])
nreps = len(data)
dist = np.tile([i + 1 for i in xrange(ndc)], (nreps, 1))
sz = [args.n_ind for i in xrange(ndc)]
sz = np.tile(np.array(sz), (nreps, 1))
# sz = [args.n_ind * args.n_markers for i in xrange(ndc)]
# sz.append(args.n_ind / 2 * args.n_markers)


nbmc = NbMC(args.mu, args.ploidy, args.nb_start,
            args.density_start, data, dist, sz,
            args.n_terms)
nbmc.set_prior_params(args.nb_mu, args.nb_tau, args.d_mu, args.d_tau)
nbmc.run_model(
    args.iter, args.burn, args.thin, args.outfile, args.plot,True)

end_time = time.time() - start_time
param.write("Run Time:" + str(end_time) + "\n")
param.close()

sz = np.array(sz, dtype=float)
pdata = np.divide(data, sz)

f = open(args.outfile + ".csv", 'r')
dist = [i + 1 for i in xrange(ndc)]

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

means = np.array(means).reshape(nreps, ndc)
upper = np.array(upper).reshape(nreps, ndc)
lower = np.array(lower).reshape(nreps, ndc)

means = np.divide(np.array(means), sz)
upper = np.divide(np.array(upper), sz)
lower = np.divide(np.array(lower), sz)

plt.clf()

fig, ax = plt.subplots(nreps, 1, sharex=True, figsize=(5, 2 * nreps))
fig.tight_layout()
for i in xrange(nreps):
    ax[i].errorbar(dist, means[i], yerr=[lower[i], upper[i]])
    ax[i].plot(dist, pdata[i], 'o')
    ax[i].set_ylim(-0.1, 1.1)
    ax[i].set_xlim(0, ndc + 1)
plt.savefig(args.outfile + str(i) + ".png")

#out = open(args.outfile + ".csv", 'a')
#out.write("Null Hypothesis DIC," + str(hoDIC) + "\n")
#out.write("Alternative Hypothesis DIC," + str(haDIC) + "\n")
#ut.write("Difference," + str(abs(hoDIC - haDIC)) + "\n")
# plt.show()
f.close()


