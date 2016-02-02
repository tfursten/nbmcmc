#!/usr/bin/env python
import argparse
import numpy as np
from Nbmcmc import *


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
    "-m", "--n_markers", required=True, type=int,
    help="number of markers")
parser.add_argument(
    "-n", "--n_ind", default=20, type=int,
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
param = open("params_" + args.outfile + ".txt", 'w')
param.write(s)
param.close()


all_data = np.array(np.genfromtxt(args.infile, delimiter=",", dtype=int))
ndc = len(all_data[0])
nreps = len(all_data) / args.n_markers
dist = np.tile([i + 1 for i in xrange(ndc)], (nreps, 1))
sz = [args.n_ind for i in xrange(ndc)]
sz = np.array(sz)
sz = np.tile(np.array(sz), (nreps, 1))


def run(mc_object, it, burn, thin, outfile, plot, rep):
    outfile = outfile + "%.3d" % rep
    mc_object.run_model(it, burn, thin, outfile, plot, True)
    #out = open(outfile + ".csv", 'a')
    #out.write("Null Hypothesis DIC," + str(hoDIC) + "\n")
    #out.write("Alternative Hypothesis DIC," + str(haDIC) + "\n")
    #out.write("Difference," + str(abs(hoDIC - haDIC)))
    # out.close()

idx = 0
for i in xrange(nreps):
    nbmc = NbMC(args.mu, args.ploidy, args.nb_start,
                args.density_start, all_data[idx:idx + args.n_markers,:],
                dist, sz, args.n_terms)
    nbmc.set_prior_params(args.nb_mu, args.nb_tau, args.d_mu, args.d_tau)
    run(nbmc, args.iter, args.burn, args.thin, args.outfile, args.plot, i)
    idx += args.nmarkers
