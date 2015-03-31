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
    if(v1<=0 || v2<=0){
        return 0xFFFFFFFFFFFFFFFF;
    }
    IBD *p = (IBD *)params;
    return -p[0].update(v1,v2);
}

int main(int argc, char *argv[])
{
    string fileName;
    if(argc != 2)
        cout << "usage: " <<argv[0] << " <filename>\n";
    else
        fileName = argv[1];
    
    
    double u = 0.0001;

    int numDataElements = 50;
    ifstream inputFile(fileName+".txt");
    if(!inputFile.is_open())
        cout << "Could not open file" << endl;
    ofstream out (fileName+"_MLENB.txt",ofstream::out);

    vector<int> sz(49,100);
    sz.push_back(50);
    vector<double> dist(numDataElements);
        
    for(int i=0; i<50; i++){
        dist[i] = i+1;
    }

    for(int line = 0; line<2000; line++){
        vector<int> data(numDataElements);
        for(int i = 0; i<numDataElements; i++)
            inputFile >> data[i];

        IBD ibd;
        ibd.initialize(u, 30, data,dist,sz);
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
        gsl_vector_set (x,0,1);
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
            status = gsl_multimin_test_size(size,0.00001);

            if(status == GSL_SUCCESS){
                printf("converged to minimum at\n");
                out << 2*3.14 * gsl_vector_get(s->x,0)*gsl_vector_get(s->x,0)*gsl_vector_get(s->x,1) << endl;

            }
            //printf("Iteration: %5d Sigma: %.5f Density %.5f f() = %7.3f size = %.3f nb = %.5f\n",
              //  iter,
                //gsl_vector_get(s->x,0),
                //gsl_vector_get(s->x,1),
                //s->fval,size, 2*3.14 * gsl_vector_get(s->x,0)*gsl_vector_get(s->x,0)*gsl_vector_get(s->x,1));
            

        }
        while(status == GSL_CONTINUE && iter < 10000);
    
        gsl_vector_free(x);
        gsl_vector_free(ss);
        gsl_multimin_fminimizer_free(s);
        
        }
    return 0;
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
		
