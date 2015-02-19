#include <gsl/gsl_multimin.h>
#include "ibd.h"




using namespace std;



//x=distance, s= sigma, u=mutation rate, fb = fbar (avg prob of IBD), a=area
//double hx(double x, double s, double u, double fb, double a){
    //return phi(x,s,u,fb,a)-
//}




int main()
{
    double u = 0.0001;
    double s = 1.0;
    double ne = 10000;
    double a = 10000;
    IBD ibd;
    ibd.initialize(u, a, 30, "data.txt");
    cout << "ibd update" << ibd.update(s,ne) << endl;
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
		
