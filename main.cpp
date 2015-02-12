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



//x=distance, s= sigma, u=mutation rate, fb = fbar (avg prob of IBD), a=area
//double hx(double x, double s, double u, double fb, double a){
    //return phi(x,s,u,fb,a)-
//}




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
    IBD ibd;
    ibd.initialize(s,m,"dist.txt",fhat,De,30);
    //double x = polylog(2,3);
    //double x = plog(0.0001 ,2);
    //double gx = 1/(4*M_PI*s*s*De);
    //double x = series(2,s,u);
    //printf("x: % .30f\n", x);
    //for(int i=0; i<6*s; i++){
      //  double x = t_series(i,s,z);
        //printf("x: % .30f\n", x);
    //}
    //for(int i=6*s; i<m; i++){
      //  double y = bessel(i,s,sqrz);
        //printf("y: % .30f\n", y);
    //}
    //printf("x: % .30f\n", x);
    return 0;
}
		
