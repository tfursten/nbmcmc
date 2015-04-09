import pymc
import pymcmc


S = pymc.MCMC(pymcmc, db='pickle')
S.sample(iter=30000,burn=5000,thin=5) #tune_interval, tune_throughout, save_interval
S.save_state()
S.trace('sigma')[:]
S.trace('density')[:]
S.trace('nb')[:]
S.trace('enb')[:]
S.trace('ed')[:]
S.sigma.summary()
S.density.summary()
S.nb.summary()
S.enb.summary()
S.ed.summary()
S.write_csv("results.csv",variables=["sigma","nb","density","enb","ed"])
pymc.Matplot.plot(S)
S.stats()
S.db.close()