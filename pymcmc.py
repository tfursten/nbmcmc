import pymc
import numpy as np 
from math import *
import sympy.mpmath as sy 
import scipy.misc as fac 
import scipy.special as sp 

f = open('./data/exp_1_tot.txt','r')



#data
data = f.readline()
data = data.strip().split("\t")
data = np.array(data,dtype=int)

sz = [100*2000 for i in xrange(49)]
sz.append(50*2000)
sz = np.array(sz)

dist = np.array([i+1 for i in xrange(50)])


#constant variables
A = 10000.0
k = 1
mu = 0.0001
z = exp(-2.0*mu)
sqrz = sqrt(1.0-z)
g0 = log(1/sqrz)
tsz = np.sum(sz)
N_dc = len(dist)
fbar = np.sum(data)/float(tsz)
plog = np.array([sy.polylog(i+1,z) for i in xrange(30)])
exp_sigma = 1.0
exp_nb = 2.0*k*pi*exp_sigma**2


def t_series(x,sigma,plog):
	sum = 0.0
	pow2 = 1
	for t in xrange(30):
		dt = 2*t
		pow2 <<= 1
		powX = 1.0
		powS = 1.0
		for i in xrange(dt):
			powX *= x
			powS *= sigma
		s = (plog[t]*powX)/(fac.factorial2(dt,exact=True)*pow2*powS)
		if((t%2)==0):
			sum += s
		else:
			sum -= s
	return sum

def bessel(x,sigma,sqrz):
	t = (x/float(sigma))*sqrz
	if(t<650):
		return sp.k0(t)
	else:
		return 0

#priors on unknown parameters
#------------Sigma & Nb -----------------------------------------------------------------------------------------------------------------------------------------------------
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
sigma   = pymc.Lognormal('sigma',   mu=log(exp_sigma), tau=1000, value=log(exp_sigma+.001))
nb      = pymc.Lognormal('nb',      mu=log(exp_nb),    tau=1000, value=log(exp_nb))

@pymc.deterministic
def es(s=sigma):
	return exp(s)

@pymc.deterministic
def enb(nb=nb):
	return exp(nb)

@pymc.deterministic
def ed(s=es,nb=enb):
	return nb/(2.0*k*s*s*pi)

#deterministic function to calculate pIBD from Wright Malecot formula
@pymc.deterministic(plot=False)
def Phi(s=es,nb=enb,g0=g0,data=data,dist=dist,sz=sz,N_dc=N_dc,tsz=tsz,fbar=fbar,plog=plog,sqrz=sqrz): #g0,data,dist,sz,N_dc,tsz,fbar,plog,sqrz,
	phi = np.zeros((N_dc))
	phi_bar = 0
	denom = nb+g0
	split = len(data)
	for i in range(N_dc):
		if dist[i]>6*s:
			split = i
			break
	for sss in xrange(split):
		if dist[sss] == 0:
			p = g0/denom
		else:
			p = t_series(dist[sss],s,plog)/denom
		phi_bar += p*sz[sss]
		phi[sss] = p
	for lll in range(split,N_dc):
		p = bessel(dist[lll],s,sqrz)/denom
		phi_bar += p*sz[lll]
		phi[lll] = p
	phi_bar /= float(tsz)

	r = (phi - phi_bar)/(1.0 - phi_bar)
	pIBD = fbar + (1.0-fbar) * r
	return pIBD
'''
#___________________________________________________________________________________________________________________________________________________________________
#___________________________________________________________________________________________________________________________________________________________________

#------------Density & Nb-------------------------------------------------------------------------------------------------------------------------------------------
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#'''
nb      = pymc.Lognormal('nb',      mu=log(exp_nb), tau=1000, value=log(exp_nb))
density = pymc.Lognormal('density', mu=0,           tau=1000,    value=0.001)

@pymc.deterministic
def enb(nb=nb):
	return exp(nb)

@pymc.deterministic
def ed(d=density):
	return exp(d)

@pymc.deterministic
def es(nb=enb,d=ed):
	return sqrt(nb/(2.0*k*pi*d))


#deterministic function to calculate pIBD from Wright Malecot formula
@pymc.deterministic(plot=False)
def Phi(nb=enb,s=es,g0=g0,data=data,dist=dist,sz=sz,N_dc=N_dc,tsz=tsz,fbar=fbar,plog=plog,sqrz=sqrz): #g0,data,dist,sz,N_dc,tsz,fbar,plog,sqrz,
	phi = np.zeros((N_dc))
	phi_bar = 0
	#denom = nb+g0
	denom = nb + g0
	'''
	split = len(data)
	for i in range(N_dc):
		if dist[i]>6*s:
			split = i
			break
	for sss in xrange(split):
		if dist[sss] == 0:
			p = g0/denom
		else:
			p = t_series(dist[sss],s,plog)/denom
		phi_bar += p*sz[sss]
		phi[sss] = p
	for lll in range(split,N_dc):
		p = bessel(dist[lll],s,sqrz)/denom
		phi_bar += p*sz[lll]
		phi[lll] = p
	'''
	for iii in xrange(N_dc):
		if dist[iii] == 0:
			p = g0/denom
		else:
			p = sy.nsum(lambda t: exp(-2*mu*t)*exp(-(dist[iii]*dist[iii])/(4.0*s*s*t))/(2.0*t), [1,sy.inf],error=False,verbose=False,method='euler-maclaurin')
			p = p/denom
		phi_bar += p*sz[iii]
		phi[iii] = p
	phi_bar /= float(tsz)

	r = (phi - phi_bar)/(1.0 - phi_bar)
	pIBD = fbar + (1.0-fbar) * r
	return pIBD
#'''
#___________________________________________________________________________________________________________________________________________________________________
#___________________________________________________________________________________________________________________________________________________________________

#------------Density & Sigma ---------------------------------------------------------------------------------------------------------------------------------------
#*******************************************************************************************************************************************************************
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
density = pymc.Lognormal('density', mu=0,              tau=1,    value=0.001)
sigma   = pymc.Lognormal('sigma',   mu=log(exp_sigma), tau=0.01, value=log(exp_sigma))

@pymc.deterministic
def ed(d=density):
	return exp(d)

@pymc.deterministic
def es(s=sigma):
	return exp(s)

@pymc.deterministic
def nb(d=ed,s=es):
	return 2*pi*s*s*d


#deterministic function to calculate pIBD from Wright Malecot formula
@pymc.deterministic(plot=False)
def Phi(s=es,d=ed,g0=g0,data=data,dist=dist,sz=sz,N_dc=N_dc,tsz=tsz,fbar=fbar,plog=plog,sqrz=sqrz): #g0,data,dist,sz,N_dc,tsz,fbar,plog,sqrz,
	phi = np.zeros((N_dc))
	phi_bar = 0
	denom = 2.0*k*pi*s*s*d+g0
	split = len(data)
	for i in range(N_dc):
		if dist[i]>6*s:
			split = i
			break
	for sss in xrange(split):
		if dist[sss] == 0:
			p = g0/denom
		else:
			p = t_series(dist[sss],s,plog)/denom
		phi_bar += p*sz[sss]
		phi[sss] = p
	for lll in range(split,N_dc):
		p = bessel(dist[lll],s,sqrz)/denom
		phi_bar += p*sz[lll]
		phi[lll] = p
	phi_bar /= float(tsz)

	r = (phi - phi_bar)/(1.0 - phi_bar)
	pIBD = fbar + (1.0-fbar) * r
	return pIBD

'''
#___________________________________________________________________________________________________________________________________________________________________
#___________________________________________________________________________________________________________________________________________________________________

#Marginal Likelihoods
Li = np.empty(N_dc, dtype=object)
Lsim = np.empty(N_dc,dtype=object)
for i in range(N_dc):
	Li[i] = pymc.Binomial('Li_%i'%i, n=sz[i], p=Phi[i], observed=True, value=data[i]) #figure out phi[i]
	Lsim[i] = pymc.Binomial('Lsim_%i'%i, n=sz[i],p=Phi[i])


