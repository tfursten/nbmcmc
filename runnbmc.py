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
    help="name of output (no extension)")
parser.add_argument(
    "--in_path", default="./", type=str,
    help="path to data"
)
parser.add_argument(
    "--out_path", default="./", type=str,
    help="path for results"
)
parser.add_argument(
    "sep", choices=['comma', 'space', 'tab', 'semicolon'],
    type=str, default='comma',
    help="set data file delimiter"
)
parser.add_argument(
    "--dist_bins", nargs='+', type=float, required=True,
    help="List of bins for distance classes or single integer"
         "represending the number of bins to be created"
)
parser.add_argument(
    "-u", "--mu", default=0.0001, type=float,
    help="mutation rate")
parser.add_argument(
    "-nb", "--nb_start", default=1.0, type=float,
    help="starting value for Neighborhood Size")
parser.add_argument(
    "-d", "--density_start", default=1.0,
    type=float, help="starting value for density")
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
    "--plot_diog", action="store_true",
    help="output diognostic plots")
parser.add_argument(
    "--plot_ppc", action="store_true",
    help="output posterior predictive check plots")
parser.add_argument(
    "--plot_prior", action="store_true",
    help="output prior and marginal posterior plot")
parser.add_argument(
    "--nb_mu", default=1.0, type=float,
    help="mean for log normal neighborhood size prior")
parser.add_argument(
    "--nb_tau", default=0.0001, type=float,
    help="precision for log normal neighborhood size prior")
parser.add_argument(
    "--d_mu", default=1.0, type=float,
    help="mean for density log normal prior")
parser.add_argument(
    "--d_tau", default=0.0001, type=float,
    help="precision for density log normal prior")
parser.add_argument(
    "--mod_comp", action="store_true",
    help="Run DIC for null and alt model")
parser.add_argument(
    "--cartesian", default=True, type=bool,
    help="Cartesian coordinates, geographical otherwise"
)
args = parser.parse_args()

start_time = time.time()

s = str("Outfile: {}{}\n"
        "Infile: {}{}\n"
        "Mu: {}\n"
        "Nb Start: {}\n"
        "Density Start: {}\n"
        "Nb Mu: {}\n"
        "Nb Tau: {}\n"
        "Density Mu: {}\n"
        "Density Tau: {}\n"
        "MCMC iterations: {}\n"
        "MCMC burn: {}\n"
        "MCMC thin: {}\n"
        "Cartesian: {}\n").format(args.out_path, args.outfile,
                                  args.in_path, args.infile,
                                  args.mu,
                                  args.nb_start,
                                  args.density_start,
                                  args.nb_mu,
                                  args.nb_tau,
                                  args.d_mu,
                                  args.d_tau,
                                  args.iter,
                                  args.burn,
                                  args.thin,
                                  args.cartesian)

sep = {'comma': ',',
       'space': ' ',
       'tab': '\t',
       'semicolon': ';'}[args.sep]
param = open(args.out_path+args.outfile + "_params.txt", 'w')
param.write(s)

# intialize model
nbmc = NbMC(args.mu, args.nb_start, args.density_start,
            args.in_path+args.infile, args.outfile, args.dist_bins,
            args.out_path, cartesian=args.cartesian, sep=sep)
# Set prior parameters
nbmc.set_prior_params(args.nb_mu, args.nb_tau, args.d_mu, args.d_tau)
# Write out distance class data
dist_info = nbmc.get_distance_classes()
param.write("Distance Bins: " + " ".join(dist_info["bins"])+"\n")
param.write("Average Distance in Bin: " +
            " ".join(dist_info["avg_dist"]) + "\n")
param.write("Avg. Pairs Per Bin: " + " ".join(dist_info["counts"]))
param.close()
dist_file_name = args.out_path + args.outfile + "_dist.txt"
np.savetxt(dist_file_name, dist_info["dist_data"],
           header="Ind1\tInd2\tDistance\tDistance Class\tAverage Distance",
           fmt='%0d %0d %.4f %0d %.4f')
# Run Model
total = nbmc.run_model(args.iter, args.burn, args.thin,
                       plot_diog=args.plot_diog,
                       plot_ppc=args.plot_ppc, plot_prior=args.plot_prior)
# Run model comparison
if args.mod_comp:
    nbmc.model_comp(args.iter, args.burn, args.thin)

end_time = time.time() - start_time
param = open(args.out_path + args.outfile + "_params.txt", 'a')
param.write("MCMC Total Values: " + str(total) + "\n")
param.write("Run Time:" + str(end_time) + "\n")
param.close()
