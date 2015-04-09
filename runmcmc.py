import pymc
import pymcmc


S = pymc.MCMC(pymcmc, db='pickle')
S.sample(iter=10000,burn=100,thin=1) #tune_interval, tune_throughout, save_interval
S.trace('sigma')[:]
S.trace('density')[:]
S.trace('nb')[:]
S.sigma.summary()
S.density.summary()
S.nb.summary()
S.write_csv("results.csv",variables=["density","sigma","nb","Lsim"])
pymc.Matplot.plot(S)


S.stats()
S.db.close()