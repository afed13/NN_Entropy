#include <iostream>
#include <fstream>
#include <getopt.h>
#include <string.h>
#include <cstdlib>
#include <iomanip>
#include <omp.h>

#include "common.h"

using namespace std;



double inversSum(int k){
  double sum = 0.0;
  if(k == 0)
    return sum;
  for(int i = 1; i <=k; i++){
    sum += (1.0 / (double)i);
  }
  return sum;
}

char* initEntropyEuc(int numNeighbor){
  double* dArray = new double[numNeighbor];
//   double dArray[numNeighbor];
  for(int i = 0; i < numNeighbor; i++){
    dArray[i] = 0.0;
  }

  return (char *)dArray;
}


void updateDistEuc(double* dArray, ANNdistArray dists, int numNeighbor){
  for(int i = 0; i < numNeighbor; i++) {
    dArray[i] += log(sqrt(dists[i]));
  }
}

void estimateEntropyEuc(double *r, int dim, int nObs, int numNeighbor){

  double c = dim / 2.0 * log(M_PI) - lgamma(dim / 2.0 + 1.0) + 
    log((double)nObs);

  for(int i = 0; i < numNeighbor; i++){
    r[i] = dim * r[i] / nObs + c - log(double(i + 1));
  }
  

  //Calculate H
  double EulerGamma = 0.577216;

  for(int i = 0; i < numNeighbor; i++){
    r[i] = r[i] - inversSum(i) + EulerGamma + log(double(i + 1));
  }
}

