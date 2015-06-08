#!/usr/bin/env python
import threading
import multiprocessing
import argparse
import numpy as np
import pymc
import concurrent.futures
from Nbmcmc import *


parser = argparse.ArgumentParser(
    description="Bayesian Estimation of Neighborhood Size "
    "using composite marginal likelihood")
parser.add_argument(
    "infile", metavar='INFILE', type=str,
    help="name of pickle file")
parser.add_argument(
    "outfile", metavar='OUTFILE', type=str,
    help="name of output file")
parser.add_argument(
    "-l", "--line", type=int, required=True,
    help="line number of data")
parser.add_argument(
    "-u", "--mu", default=0.0001, type=float,
    help="mutation rate")
parser.add_argument("-k", "--ploidy", default=1.0, type=float,
                    help="ploidy")
parser.add_argument(
    "-s", "--sigma_start", default=1.0, type=float,
    help="starting value for sigma")
parser.add_argument("-d", "--density_start", default=1.0,
                    type=float, help="starting value for density")
parser.add_argument("-n", "--n_reps", default=10,
                    type=int, help="number of replicate datasets")
parser.add_argument(
    "-it", "--iter", default=10000, type=int,
    help="number of MCMC iterations")
parser.add_argument(
    "-b", "--burn", default=100, type=int,
    help="length of burnin period for MCMC")
parser.add_argument(
    "-nt", "--n_terms", default=30, type=int,
    help="number of terms for Taylor Series")
parser.add_argument(
    "-th", "--thin", default=1, type=int,
    help="thin the MCMC chains")
parser.add_argument(
    "--max", default=20, type=int,
    help="maximum number of worker threads")
parser.add_argument(
    "-ait", "--adj_iter", default=10000, type=int,
    help="number of adjustment MCMC iterations")
parser.add_argument(
    "-ab", "--adj_burn", default=100, type=int,
    help="length of burnin period for adjustment MCMC")
parser.add_argument(
    "-at", "--adj_thin", default=1, type=int,
    help="thin the adjustment MCMC chains")
parser.add_argument(
    "--dist", default=50, type=int,
    help="number of distance classes")
parser.add_argument(
    "-p", "--plot", action="store_true",
    help="output plots")

# we need name of original file
args = parser.parse_args()
mcmctot = args.iter / args.thin
s = str("Outfile: {}\nInfile: {}\nMu: {}\nPloidy: {}\n"
        "Sigma Start: {}\nDensity Start: {}\n"
        "Taylor Series Terms: {}\nMCMC iterations: {}\n"
        "MCMC burn: {}\nMCMC thin: {}\n"
        "MCMC total: {}").format(args.outfile, args.infile, args.mu,
                                 args.ploidy, args.sigma_start,
                                 args.density_start, args.n_terms,
                                 args.iter, args.burn, args.thin, mcmctot)
print(s)
data_file = open(args.infile, 'r')

# fine line number of data
for line in xrange(args.line):
    data_file.readline()

# Run model
data = data_file.readline().strip().split(",")
data = np.array(data, dtype=int)
dist = np.array([i + 1 for i in xrange(len(data))])
sz = np.array([100 for i in xrange(len(data))])
nbmc = NbMC(args.mu, args.ploidy, args.sigma_start,
            args.density_start, data, dist, sz,
            args.n_terms)
nbmc.run_model(
    args.iter, args.burn, args.thin, args.outfile + str(args.line), args.plot)

# reopen trace database
db = pymc.database.pickle.load(args.outfile + str(args.line) + ".pickle")

rep_trace = np.empty((args.dist, args.n_reps))

for i in xrange(args.dist):
    # get last n from trace for each distance class
    rep_trace[i] = np.array(db.trace('Lsim_%i' % i)[-args.n_reps:])
# this is data generated from the model using estimated parameters
rep_trace = rep_trace.transpose()


reps = np.array([NbMC(args.mu, args.ploidy, args.sigma_start,
                      args.density_start, rep_trace[i], dist, sz,
                      args.n_terms) for i in xrange(args.n_reps)],
                dtype=object)


def run(mc_object, it, burn, thin, outfile, plot, rep):
    outfile = outfile + str(args.line) + "_" + str(rep)
    mc_object.run_model(it, burn, thin, outfile, plot)

with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_reps) as executor:
    future_to_run = {executor.submit(run, reps[thr], args.adj_iter,
                                     args.adj_burn, args.adj_thin,
                                     args.outfile, args.plot,
                                     thr): thr for
                     thr in xrange(args.n_reps)}
    for future in concurrent.futures.as_completed(future_to_run):
        thr = future_to_run[future]

adj_sigma = np.empty(0)
adj_density = np.empty(0)
adj_nb = np.empty(0)

for i in xrange(args.n_reps):
    db = pymc.database.pickle.load(
        args.outfile + str(args.line) + "_" + str(i) + ".pickle")
    s = np.array(db.trace('sigma')[:])
    d = np.array(db.trace('density')[:])
    n = np.array(db.trace('enb')[:])
    adj_sigma = np.append(adj_sigma, s)
    adj_density = np.append(adj_density, d)
    adj_nb = np.append(adj_nb, n)

adj_sigma = np.sort(adj_sigma)
adj_density = np.sort(adj_density)
adj_nb = np.sort(adj_nb)

out = open("Adj_" + args.outfile + str(args.line) + ".txt", 'w')
out.write("nb,sigma,density\n")
for n, s, d in zip(adj_nb, adj_sigma, adj_density):
    out.write(str(n) + "," + str(s) + "," + str(d) + "\n")
out.close()
