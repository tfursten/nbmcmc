import pymc
import numpy as np 
from math import *
import sympy.mpmath as sy 
import scipy.misc as fac 
import scipy.special as sp 

f = open('test.txt','r')



#data
data = f.readline()
data = data.strip().split(" ")
data = np.array(data,dtype=int)

sz = [100 for i in xrange(49)]
sz.append(50)
sz = np.array(sz)

dist = np.array([i+1 for i in xrange(50)])


#constant variables
A = 10000
k = 1
mu = 0.0001
z = exp(-2.0*mu)
sqrz = sqrt(1.0-z)
g0 = log(1/sqrz)
tsz = np.sum(sz)
N_dc = len(dist)
fbar = np.sum(data)/float(tsz)
plog = np.array([sy.polylog(i+1,z) for i in xrange(30)])



#priors on unknown parameters
sigma = pymc.Lognormal('sigma',mu=0,tau=0.01,value=2)
density = pymc.Lognormal('density', mu=1, tau=5,value=1)

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


#deterministic function to calculate pIBD from Wright Malecot formula
@pymc.deterministic(plot=False)
def Phi(s=sigma,d=density,g0=g0,data=data,dist=dist,sz=sz,N_dc=N_dc,tsz=tsz,fbar=fbar,plog=plog,sqrz=sqrz): #g0,data,dist,sz,N_dc,tsz,fbar,plog,sqrz,
	'''return a vector phi'''
	ss=s
	s=sqrt(ss)
	phi = np.zeros((N_dc))
	phi_bar = 0
	denom = k*pi*ss*d*2.0+g0
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




#Marginal Likelihoods
Li = np.empty(N_dc, dtype=object)
Lsim = np.empty(N_dc,dtype=object)
for i in range(N_dc):
	Li[i] = pymc.Binomial('Li_%i'%i, n=data[i], p=Phi[i], observed=True, value=0.5) #figure out phi[i]
	Lsim[i] = pymc.Binomial('Lsim_%i'%i, n=data[i],p=Phi[i])
@pymc.deterministic
def nb(s=sigma,d=density):
	return 2*pi*s*d