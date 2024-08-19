#include <iostream>
#include <fstream>
#include "ANN/ANN.h"
#include <omp.h>

using namespace std;

char * initEntropyEuc(int);
void updateDistEuc(double *, ANNdistArray, int);
void estimateEntropyEuc(double *, int, int, int);
void estimateEntropyEucTest(double *, int, int, int);
