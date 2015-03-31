#include "ibd.h"

void IBD::initialize(double mu, int num_t_series_terms, vector<int> & dataIn, vector<double> &distIn, vector<int> &szIn){
	u = mu;
    ntt = num_t_series_terms;
	z = exp(-2*u);
	sqrz = sqrt(1-z);
    data = dataIn;
    sz = szIn;
    dist = distIn;
    int f = accumulate(data.begin(),data.end(),0);
    tsz  = accumulate(sz.begin(),sz.end(),0);
    fhat = f/tsz;
    ndc = dist.size();
    g0 = log(1/sqrz);
    for(int t=0; t<ntt; t++)
        plog.push_back(polylog(t+1,z));
    split = data.size();
}

double IBD::update(double sigma_i, double de_i){
    s = sigma_i;
    ss = s*s;
    de = de_i;
    //g0 = t_series(0);
    
    for(int i=0; i<ndc; i++){
        if(dist[i]>6*s){
            split = i;
            break;
        }
    }
    
    return cml();
}





double IBD::cml(){
    vector<double> phi(ndc,0);
    double phi_bar=0;
    double denom = 2*ss*M_PI*de+g0;
    double p;
    
    for(int sss=0; sss<split; sss++){
        if(dist[sss]==0)
            p = g0/denom;
        else
            p = t_series(dist[sss])/denom;
        phi_bar += p*sz[sss];
        phi[sss] = p;
    }
    for(int lll=split; lll<ndc; lll++){
        p = bessel(dist[lll])/denom;
        phi_bar += p*sz[lll];
        phi[lll] = p;
    }
    
    phi_bar /= tsz;
    double cml = 0;
    for(int i=0; i<ndc; i++){
        double r = (phi[i] - phi_bar)/(1.0-phi_bar);
        double pIBD = fhat + (1-fhat) * r;
        //cout << "pIBD: " << pIBD << " Data: " << data[i]/float(sz[i]) << endl;
        cml += data[i]*log(pIBD)+(sz[i]-data[i])*log(1-pIBD);
    }
    //cout << cml << endl;
    return cml;
}

double IBD::t_series(double x){
	assert(ntt<64);
    double sum = 0.0;
    int64_t pow2 = 1; 
    int dt;
    double powX, powS;
    for(int t=0; t<ntt; t++){
        dt = 2*t;
        pow2 <<= 1;
        powX = 1.0;
        powS = 1.0;
        for(int i=0; i<dt; i++){
            powX *= x;
            powS *= s;
        }
        double tt = (plog[t]*powX)/(gsl_sf_doublefact(dt)*pow2*powS);
        if(t%2==0)
            sum += tt;
        else
            sum -= tt;
    }
	return sum;
}

double IBD::bessel(double x){
    double t = (x/s)*sqrz;
    if(t<650){
        return gsl_sf_bessel_K0(t);
    }
    else
        return 0;
}