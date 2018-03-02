/*==========================================================
 * Multrnd_Matrix_mex.c - 
 *
 *
 * The calling syntax is:
 *
 *		[ZSDS,WSZS] = Multrnd_Matrix_mex_fast(Xtrain,Phi,Theta);
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2012 Mingyuan Zhou
 *
 *========================================================*/
/* $Revision: 0.1 $ */

#include "mex.h"
/* //#include <math.h>
//#include <stdlib.h>
//#include <stdio.h>
//#include "matrix.h"*/
#include "cokus.c"
#define RAND_MAX_32 4294967295.0

/* //  The computational routine 
//void Multrnd_Matrix(double *ZSDS, double *WSZS, double *Phi, double *Theta, mwIndex *ir, mwIndex *jc, double *pr, mwSize Vsize, mwSize Nsize, mwSize Ksize, double *RND, double *prob_cumsum) //, mxArray **lhsPtr, mxArray **rhsPtr)*/


mwIndex BinarySearch(double probrnd, double *prob_cumsum, mwSize Ksize) {
    mwIndex k, kstart, kend;
    if (probrnd <=prob_cumsum[0])
        return(0);
    else {
        for (kstart=1, kend=Ksize-1; ; ) {
            if (kstart >= kend) {
                /*//k = kend;*/
                return(kend);
            }
            else {
                k = kstart+ (kend-kstart)/2;
                if (prob_cumsum[k-1]>probrnd && prob_cumsum[k]>probrnd)
                    kend = k-1;
                else if (prob_cumsum[k-1]<probrnd && prob_cumsum[k]<probrnd)
                    kstart = k+1;
                else
                    return(k);
            }
        }
    }
    return(k);
}

void Multrnd_Matrix(double *ZSDS, double *WSZS, double *Phi, double *Theta, mwIndex *ir, mwIndex *jc, double *pr, mwSize Vsize, mwSize Nsize, mwSize Ksize,  double *prob_cumsum) 
/*//, mxArray **lhsPtr, mxArray **rhsPtr)*/
{    
  
    double cum_sum, probrnd;
    mwIndex k, j, v, token, ji=0, total=0; 
	/*//, ksave;*/
    mwIndex starting_row_index, stopping_row_index, current_row_index;
    
    
    for (j=0;j<Nsize;j++) {
        starting_row_index = jc[j];
        stopping_row_index = jc[j+1];
        if (starting_row_index == stopping_row_index)
            continue;
        else {
            for (current_row_index =  starting_row_index; current_row_index<stopping_row_index; current_row_index++) {
                v = ir[current_row_index];                
                for (cum_sum=0,k=0; k<Ksize; k++) {
                    /*//prob[k] = Phi(v+ k*Vsize)*Theta(k + Ksize*i);*/
                    cum_sum += Phi[v+ k*Vsize]*Theta[k + Ksize*j];                    
                    prob_cumsum[k] = cum_sum;
                }
                /*// mexPrintf("Rows = %d; Columns = %d; total = %d; value = %d\n", v+1, j+1, total, (mwSize)  pr[total]);*/
                for (token=0;token< pr[total];token++) {
                    /*//probrnd = RND[ji]*cum_sum;*/
                    probrnd = (double) randomMT()/RAND_MAX_32*cum_sum;
                  /* // probrnd = rand()*cum_sum./MAX_RAND;
                  //  probrnd = (double) cum_sum * (double) randomMT() / (double) 4294967296.0;
                //    mexCallMATLAB(1, lhsPtr, 1,  rhsPtr, "rand");
                //  mexPrintf("%f\n",drand48());
                //    probrnd =  *mxGetPr(lhsPtr[0]) *cum_sum;*/
                    
                    ji++;
                    
/*//                     for (k=0;k<Ksize;k++) {
//                         if (prob_cumsum[k]>=probrnd)
//                             break;
//                     }
//                     
//                     ksave = k;*/
                            
                    k = BinarySearch(probrnd, prob_cumsum, Ksize);   
                    
                    /*//if(ksave!=k)
                    //    mexPrintf("%d,%d\n",ksave,k);*/
                    ZSDS[k+Ksize*j]++;
                    WSZS[v+k*Vsize]++;
                }
                total++;
            }
        }
    }
   /*// mexPrintf("total=%d, Ji = %d",total,ji);*/
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *ZSDS, *WSZS, *Phi, *Theta, *RND;
    double  *pr, *prob_cumsum;
    mwIndex *ir, *jc;
    mwIndex Vsize, Nsize, Ksize;
    
/*//         mxArray *lhs, *rhs;
//         double *rhsPtr, *lhsPtr;

//  mwIndex      row, col, total=0, number_of_columns;
//  mwIndex      starting_row_index, stopping_row_index, 
// current_row_index;    
// */    
     pr = mxGetPr(prhs[0]);
     ir = mxGetIr(prhs[0]);
     jc = mxGetJc(prhs[0]);        
     Vsize = mxGetM(prhs[0]);
     Nsize = mxGetN(prhs[0]);
     Ksize = mxGetN(prhs[1]);
     Phi = mxGetPr(prhs[1]);
     Theta = mxGetPr(prhs[2]);
    /*// RND = mxGetPr(prhs[3]);*/
    
    plhs[0] = mxCreateDoubleMatrix(Ksize,Nsize,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(Vsize,Ksize,mxREAL);
    ZSDS = mxGetPr(plhs[0]);
    WSZS = mxGetPr(plhs[1]);
    
    prob_cumsum = (double *) mxCalloc(Ksize,sizeof(double));
    

/*//     rhs =  mxCreateDoubleMatrix(1,1,mxREAL);
//     lhs =  mxCreateDoubleMatrix(1,1,mxREAL);
//    mexCallMATLAB(0,NULL,1,&rhs, "disp");
//     
//     rhsPtr = mxGetPr(rhs);
//     rhsPtr[0]=1;
    
//    number_of_columns = mxGetN(prhs[0]);
//    for (col=0; col<number_of_columns; col++)  { 
//       starting_row_index = jc[col]; 
//       stopping_row_index = jc[col+1]; 
//       if (starting_row_index == stopping_row_index)
//          continue;
//       else {
//          for (current_row_index = starting_row_index; 
//               current_row_index < stopping_row_index; 
//               current_row_index++) 
//             mexPrintf("(%d,%d) = %g\n", ir[current_row_index]+1, 
//                                         col+1, pr[total++]);
//       }
//    }  
*/
/*   //Multrnd_Matrix(ZSDS, WSZS, Phi, Theta, ir, jc, pr,  Vsize, Nsize, Ksize, RND, prob_cumsum); //, &lhs, &rhs); */
   
   Multrnd_Matrix(ZSDS, WSZS, Phi, Theta, ir, jc, pr,  Vsize, Nsize, Ksize,  prob_cumsum); 
   /*//, &lhs, &rhs); */
}