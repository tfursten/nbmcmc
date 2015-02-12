#include "ibd.h"

void IBD::initialize(double sigma, double mu, double nc, double f, double d){
	s = sigma;
	ss = 6*sigma;
	u = mu;
	ndc = nc;
	fhat = f;
	de = d;
	z = exp(-2*u);
	sqrz = sqrt(1-z);
	g0 = t_series(0,s,z);
}

double IBD::t_series(double x, N=30){
	assert(N<64);
    double sum = 0.0;
    int64_t pow2 = 1; 
	for(int t=0; t<N; t++){
        int dt = 2*t;
        pow2 <<= 1;
        double powX = 1.0;
        double powS = 1.0;
        for(int i=0; i<dt; i++){
            powX *= x;
            powS *= s;
        }
        double s = (polylog(t+1,z)*powX)/(gsl_sf_doublefact(dt)*pow2*powS);
        if(t%2==0)
            sum += s;
        else
            sum -= s;
    }
	return sum;
}

double IBD::bessel(double x){
	return gsl_sf_bessel_K0((x/s)*sqrz);
}