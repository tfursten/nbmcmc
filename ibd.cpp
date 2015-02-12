#include "ibd.h"

void IBD::initialize(double sigma, double mu, string dist_file, double fhat, double density, int terms){
	s = sigma;
	ss = s*s;
	u = mu;
	f = fhat;
	de = density;
    N = terms;
	z = exp(-2*u);
	sqrz = sqrt(1-z);
	g0 = t_series(0);
    ifstream infile(data_file);
    while(!infile.eof()){
        double dis;
        int ct, szof;
        infile >> dis >> ct >> szof;
        data[dis] = make_pair(ct,szof);
        tsz += szof;
    }
    split = data.size();
    for(datamap::iterator it = data.begin(), int i = 0; it != data.end(); ++it, i++){
        cout << it->first << " " << it->second.first << " " it->second.second << endl;
        if((i->first)>6*s){
            split = i;
            break;
        }
    }
    cout << split << endl;
}



double IBD::cml(){
    double p = 1;
    func = &IBD::t_series;
    for(int s=0; s<split; s++){
        p*=pibd(dist[s]);
    }
    func = &IBD::bessel;
    for(int l=split; l<data.size(); l++){
        p*=pibd(dist[l]);
    }
}
        

void IBD::phi(){
    vector<double> phi;
    double phi_bar;
    for(datamap::iterator it = data.begin(); it != data.end(); ++it){
        double p = (this->*func)(it->first)/(2*ss*M_PI*de+g0);
        phi.insert(p);
        phi_bar += p*it->second.second; 
    }
    phi_bar = phi_bar/tsz;

}

double IBD::hx(double x){
    return phi(x)-phat()
}
    


double IBD::t_series(double x){
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