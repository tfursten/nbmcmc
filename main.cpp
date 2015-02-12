#include <iostream>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <assert.h>
#include "polylog/polylog.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include "ibd.h"




using namespace std;
//For mu 10^-4
static double PLOG[30] = {
8.5172931897496280,
1.6430306182100543,
1.2017281166907554,
1.0820828552153867,
1.0367113145355733,
1.0171356980782826,
1.0081458295066380,
1.0038757066879467,
1.0018075975204717,
1.0007941935294555,
1.0002940097279232,
1.0000460077341429,
0.9999226841388169,
0.9998612435959770,
0.9998305959878008,
0.9998152961416530,
0.9998076541404645,
0.9998038357647978,
0.9998019274480775,
0.9998009735791344,
0.9998004967408992,
0.9998002583538018,
0.9998001391709121,
0.9998000795830166,
0.9998000497902509,
0.9998000348942621,
0.9998000274463988,
0.9998000237225109,
0.9998000218605815,
0.9998000209296217,
};


//x=distance, s= sigma, u=mutation rate, fb = fbar (avg prob of IBD), a=area
//double hx(double x, double s, double u, double fb, double a){
    //return phi(x,s,u,fb,a)-
//}


double t_series(double x, double s, double z, int N=30){
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


double bessel(double x, double s, double sqrz){
    return gsl_sf_bessel_K0((x/s)*sqrz);
}

int main()
{
    int m = 5;
    double u = 0.0001;
    double s = 2;
    double De = 1;
    double fhat = .33;
    double theta = 2.0;
    double *arr = new double[m];
    double z = exp(-2*u);
    double sqrz = sqrt(1-z);
    //double x = polylog(2,3);
    //double x = plog(0.0001 ,2);
    //double gx = 1/(4*M_PI*s*s*De);
    //double x = series(2,s,u);
    //printf("x: % .30f\n", x);
    for(int i=0; i<6*s; i++){
        double x = t_series(i,s,z);
        printf("x: % .30f\n", x);
    }
    for(int i=6*s; i<m; i++){
        double y = bessel(i,s,sqrz);
        printf("y: % .30f\n", y);
    }
    //printf("x: % .30f\n", x);
    return 0;
}
		
