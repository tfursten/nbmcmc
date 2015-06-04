#!/usr/bin/env python
import threading
import argparse
import numpy as np
import concurrent.futures
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
parser.add_argument("-k", "--ploidy", default=1.0, type=float,
                    help="ploidy")
parser.add_argument(
    "-s", "--sigma_start", default=1.0, type=float,
    help="starting value for sigma")
parser.add_argument("-d", "--density_start", default=1.0,
                    type=float, help="starting value for density")
parser.add_argument("-t", "--n_terms", default=30,
                    type=int, help="number of terms for taylor series")
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
    "--max", default=20, type=int,
    help="maximum number of worker threads")

args = parser.parse_args()
# TODO parse different types of data files
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
param = open("parameters.txt", 'w')
param.write(s)
param.close()

'''
data = np.array(np.genfromtxt(args.infile, delimiter=",", dtype=int))
data = data[0]
dist = np.array([i + 1 for i in xrange(len(data))])
sz = np.array([100 for i in xrange(len(data))])
nbmc = NbMC(args.mu, args.ploidy, args.sigma_start,
            args.density_start, data, dist, sz,
            args.n_terms)
nbmc.run_model(args.iter, args.burn, args.thin, args.outfile, args.plot)

'''
data = np.array(np.genfromtxt(args.infile, delimiter=",", dtype=int))
dist = np.array([i + 1 for i in xrange(len(data[0]))])
sz = np.array([100 for i in xrange(len(data[0]))])


reps = np.array([NbMC(args.mu, args.ploidy, args.sigma_start,
                      args.density_start, data[i], dist, sz,
                      args.n_terms) for i in xrange(len(data))], dtype=object)


def run(mc_object, it, burn, thin, outfile, plot, rep):
    outfile = outfile + str(rep)
    mc_object.run_model(it, burn, thin, outfile, plot)


with concurrent.futures.ThreadPoolExecutor(max_workers=args.max) as executor:
    future_to_run = {executor.submit(run, reps[thr], args.iter,
                                     args.burn, args.thin,
                                     args.outfile, args.plot,
                                     thr): thr for
                     thr in xrange(len(reps))}
    for future in concurrent.futures.as_completed(future_to_run):
        thr = future_to_run[future]


'''
threads = []
for thr in xrange(len(data)):
    t = threading.Thread(target=run,
                         args=(reps[thr], args.iter,
                               args.burn, args.thin,
                               args.outfile, args.plot, thr))
    threads.append(t)
    t.start()
'''
