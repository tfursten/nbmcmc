#include <gsl/gsl_multimin.h>
#include "ibd.h"


#include <iostream>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "polylog/polylog.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>

using namespace std;



//x=distance, s= sigma, u=mutation rate, fb = fbar (avg prob of IBD), a=area
//double hx(double x, double s, double u, double fb, double a){
    //return phi(x,s,u,fb,a)-
//}


double my_fun(const gsl_vector *v, void *params){
    double v1 = gsl_vector_get(v,0);
    double v2 = gsl_vector_get(v,1);
    cout << v1 << " " << v2 << endl;
    if(v1<=0 || v2<=0)
        return 0xFFFFFFFFFFFFFFFF;
    IBD *p = (IBD *)params;
    return p[0].update(v1,v2);
}

int main()
{
  

    double u = 0.0001;
    double a = 10000;

    IBD ibd;
    ibd.initialize(u, a, 30, "data.txt");
    IBD par[1] = {ibd};
    const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex;
    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *ss, *x;
    gsl_multimin_function minex_func;

    size_t iter = 0;
    int status; 
    double size;

    /*starting point */
    x = gsl_vector_alloc (2);
    gsl_vector_set (x,0,.5);
    gsl_vector_set (x,1,1.5);

    /*set initial step sizes to 1*/
    ss = gsl_vector_alloc(2);
    gsl_vector_set_all(ss,1.0);

    /*Initialize method and iterate*/
    minex_func.n = 2;
    minex_func.f = my_fun;
    minex_func.params = par;

    s = gsl_multimin_fminimizer_alloc(T,2);
    gsl_multimin_fminimizer_set(s,&minex_func,x,ss);

    do{
        iter++;
        status = gsl_multimin_fminimizer_iterate(s);
        if(status)
            break;
        size = gsl_multimin_fminimizer_size(s);
        status = gsl_multimin_test_size(size,0.1);

        if(status == GSL_SUCCESS){
            printf("converged to minimum at\n");
        }
        printf("%5d %.5f %.5f f() = %7.3f size = %.3f\n",
            iter,
            gsl_vector_get(s->x,0),
            gsl_vector_get(s->x,1),
            s->fval,size);
    }
    while(status == GSL_CONTINUE && iter < 10000);
    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);
    return status;
}

    //cout << "ibd update" << ibd.update(s,ne) << endl;
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
   // return 0;
//}
		
