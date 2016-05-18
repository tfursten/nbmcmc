#!/bin/bash
iter="20000"
burn="0"
thin="1"
outfile="results"
nb_tau="0.0001"
nb_mu="2"
d_tau="0.0001"
density="1.0"
ind="20"
nstart="2"
dstart="2"
markers="5"
parallel nohup ./runnbmc.py ./sample_data/sim/ibd/m{1}_ibd.txt ./genData/results/results_{1} -it {3} -b {4} -th {5} -n {6} --nb_mu {7} --d_tau {8} --nb_tau {9} --nb_start {10} --density_start {11} ::: $markers ::: $infile ::: $iter ::: $burn ::: \
$thin ::: $ind ::: $nb_mu ::: $d_tau ::: $nb_tau ::: $nstart ::: $dstart &

