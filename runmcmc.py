import pymc
import pymcmc


S = pymc.MCMC(pymcmc, db='pickle', calc_deviance=False)
S.sample(iter=30,burn=1,thin=1) #tune_interval, tune_throughout, save_interval
S.save_state()
S.trace('es')[:]
S.trace('ed')[:]
S.trace('enb')[:]
S.es.summary()
S.ed.summary()
S.enb.summary()
S.write_csv("results.csv",variables=["es","enb","ed"])
pymc.Matplot.plot(S)
S.stats()
S.db.close()