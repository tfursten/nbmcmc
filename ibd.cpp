#include "ibd.h"

void IBD::initialize(double mu, double area, int num_t_series_terms, string data_file){
	u = mu;
    a = area;
    ntt = num_t_series_terms;
	z = exp(-2*u);
	sqrz = sqrt(1-z);
    ifstream infile(data_file);
    int f = 0;
    ndc = 0;
    while(!infile.eof()){
        ndc += 1;
        double d;
        int ct, t;
        infile >> d >> ct >> t;
        dist.push_back(d);
        data.push_back(ct);
        sz.push_back(t);
        tsz += t;
        f += ct;
    }
    fhat = f/float(tsz);
    cout << "fhat: " << fhat << endl;
}

double IBD::update(double sigma_i, double ne_i){
    s = sigma_i;
    ss = s*s;
    ne = ne_i;
    de = ne/a;
    g0 = t_series(0);
    split = data.size();
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
    for(int s=0; s<split; s++){
        double p = t_series(dist[s])/denom;
        phi_bar += p*sz[s];
        phi[s] = p;
    }
    for(int l=split; l<ndc; l++){
        double p = bessel(dist[l])/denom;
        phi_bar += p*sz[l];
        phi[l] = p;
    }
    phi_bar /= tsz;
    double cml = 0;
    for(int i=0; i<ndc; i++){
        double r = (phi[i] - phi_bar)/(1-phi_bar);
        double pIBD = fhat + (1-fhat) * r;
        cml += data[i]*log(pIBD)+(sz[i]-data[i])*log(1-pIBD);
    }
    cout << cml << endl;
    return cml;
}

double IBD::t_series(double x){
	assert(ntt<64);
    double sum = 0.0;
    int64_t pow2 = 1; 
	for(int t=0; t<ntt; t++){
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