  
// Program nn_entropy
// multithreaded entropy program
// OpenMP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <cstdlib>
#include <string>
#include <string.h>
#include <vector>

#include "common.h"
#include "nn_entropy.h"

using namespace std;


typedef double	TORS;			// coordinate data type
typedef TORS* TORSpoint;		// a point
typedef TORSpoint* TORSpointArray;	// an array of points 

typedef int CLUS;
typedef CLUS* CLUSpoint;
typedef CLUSpoint* CLUSpointArray;

int getClusNumber(int* minpoint,int *maxpoint,int maxclus,int nvar);
int isTheLast(int* apoint,int* lpoint,int asize,int lsize);
int* getNextCluster2(int* cpoint, int nvar,int maxclus);
int* shiftNextCluster2(int* cpoint, int nvar,int maxclus,int posit);

const ANNsplitRule def_split		= ANN_KD_SUGGEST;	// def splitting rule
				

int main(int ARGC, char **ARGV) 
{

    if(ARGC <= 1) {
	cout << "Usage: ./nn_entropy.exe nproc < control_file " << endl;
	exit(0);
    }

# ifdef _OPENMP
    int act_proc_num = omp_get_num_procs();
# endif

    char comment[150];
    char getval_f[150];
    char v_opt[2];
    cin.get(comment,150).get();
    cin.get(v_opt,2).get();
    int v_in = strcmp(v_opt,"v");

// getting arguments

    int n_proc = atoi(ARGV[1]);    // number of CPUs to use

    char dfile[150];               // datafile
    cin.get(comment,150).get();
    cin.get(dfile,150).get();
    cin.get(comment,150).get();
    cin.get(getval_f,150).get();
    int f_point = atoi(getval_f);              // first point
    cin.get(comment,150).get();
    cin.get(getval_f,150).get();
    int nrec = atoi(getval_f);     // number of records
    cin.get(comment,150).get();
    cin.get(getval_f,150).get();
    int nvar = atoi(getval_f);      // number of variables
    cin.get(comment,150).get();
    cin.get(getval_f,150).get();
    int k_nn    = atoi(getval_f);         // k-NN
    cin.get(comment,150).get();
    cin.get(getval_f,150).get();
    double epsilon = atof(getval_f);      // epsilon
    cin.get(comment,150).get();
    cin.get(getval_f,150).get();
    int minclus = atoi(getval_f);         // minimum clustersize
    cin.get(comment,150).get();
    cin.get(getval_f,150).get();
    int maxclus = atoi(getval_f);         // maximum clustersize

    //printouts for verbose mode
    if(v_in == 0) {
      cout << "\n\n\t Program ANN_ENTROPY v4.3" <<endl;
      cout << "Datafile:            " << dfile << endl;
      cout << "First point:         " << f_point << endl;
      cout << "Number of Records:   " << nrec << endl;
      cout << "Number of Variables: " << nvar << endl;
      cout << "Number of requested CPUs:      " << n_proc << endl;
# ifdef _OPENMP
      cout << "Number of available CPUs:      " << act_proc_num << endl;
# endif
      cout << "MinClus:             " << minclus << endl;
      cout << "MaxClus:             " << maxclus << endl;
      cout << "k-NN:                " << k_nn << endl;        
      cout << "Epsilon:             " << epsilon << endl;   
    }
    
    cout << "Data size: " << nrec << ", dimension: "<< nvar << endl;


// restriction checks

    if(nvar <= 0 || nvar > MAX_VAR) {
	cout << "Invalid value of nvar: " << nvar << endl;
	cout << "0 < nvar <= "<< MAX_VAR<< endl;
	exit(0);
    }
    if(nrec <= 0) {
	cout << "Invalid value of nrec: " << nrec << endl;
	exit(0);
    }
    if(n_proc < 0) {
	cout << "Invalid value of n_proc: " << n_proc << endl;
	exit(0);
    }
# ifdef _OPENMP
    if(n_proc > act_proc_num) {
	cout << "Invalid value of n_proc: " << n_proc << endl;
	exit(0);
    }
# endif
     
    if(minclus <= 0 || minclus > MAX_CLUSTERSIZE) {
	cout << "Invalid value of minclus: " << minclus << endl;
	cout << "0 < minclus <= "<< MAX_CLUSTERSIZE<< endl;
	exit(0);
    }
    if(maxclus <= 0 || maxclus > MAX_CLUSTERSIZE) {
	cout << "Invalid value of maxclus: " << maxclus << endl;
	cout << "0 < maxclus <= "<< MAX_CLUSTERSIZE<< endl;
	exit(0);
    }
    if(minclus > maxclus) {
	cout << "minclus should be smaller than or equal to maxclus" << endl;
	cout << "\tminclus = " << minclus << "\t\t maxclus = "<< maxclus << endl;
	exit(0);
    }
    if(k_nn <= 0 || k_nn > MAX_KNN) {
	cout << "Invalid value of k-NN: " << k_nn << endl;
	cout << "0 < k-NN <= "<< MAX_KNN<< endl;
	exit(0);
    }
    if(epsilon < 0 || epsilon > MAX_EPSILON) {
	cout << "Invalid value of epsilon: " << epsilon << endl;
	cout << "0 < epsilon <= "<< MAX_EPSILON<< endl;
	exit(0);
    }

// Checking starting points and endpoints
    int * minpoint = new int[maxclus+1];
    int * maxpoint = new int[maxclus+1];
    if(minpoint == NULL) exit(0);
    if(maxpoint == NULL) exit(0);


//setting empty elements for minclus smaller than maxclus
    for(int i=0;i<=maxclus;i++) minpoint[i]=0;     
    cin.get(comment,150).get();                 // comment line

    for(int i=1;i<=minclus;i++) {   // changed fom maxclus
      int iii = maxclus-minclus;
      int n_colclus = 0;
      cin.get(getval_f,150).get();
      n_colclus = atoi(getval_f);
      minpoint[iii+i] = n_colclus;
      if(minpoint[iii+i] < 1 || minpoint[iii+i] > nvar) {
	cout << "Invalid value of minclus restart point ("<<iii+1<<"):" <<minpoint[iii+i]<< endl ;
	for(int iz=0;iz<=maxclus;iz++) cout << iz << "\t" <<minpoint[iz] <<endl;
	cout << iii+i << "\t"<< n_colclus  <<endl;
	exit(0);
      }
      if(i>1) {
	if(minpoint[iii+i]<=minpoint[iii+i-1]) {
	  cout << "Invalid value of minclus restart point ("<<iii+i<<"):" <<minpoint[iii+i]<< endl ;
	  cout << "It should not be smaller/equal that the previous value." << endl; 
	  for(int iz=0;iz<=maxclus;iz++) cout << iz << "\t" <<minpoint[iz] <<endl;
	  exit(0);
	}
      }
    }
    minpoint[0]=minclus;

    // verbose output
    if(v_in == 0) {
      cout << "minpoint:" <<endl;
      for(int i=0;i<=maxclus;i++) cout << i << "\t" <<minpoint[i] <<endl;
    }

    cin.get(comment,150).get();                 // comment line
    for(int i=1;i<=maxclus;i++) {
      int n_colclus = 0;
      cin.get(getval_f,150).get();
      n_colclus = atoi(getval_f);
      maxpoint[i]=n_colclus;
      if(maxpoint[i] < 1 || maxpoint[i] > nvar) {
	cout << "Invalid value of maxclus restart point ("<<i+1<<"):" <<maxpoint[i]<< endl ;
	exit(0);
      }
      if(i>1) {
	if(maxpoint[i]<=maxpoint[i-1]) {
	  cout << "Invalid value of maxclus restart point ("<<i+1<<"):" <<maxpoint[i]<< endl ;
	  cout << "It should not be smaller/equal that the previous value." << endl; 
	  exit(0);
	}
      }
    }
    maxpoint[0]=maxclus;

    // verbose output
    if(v_in == 0) {
      cout << "maxpoint:" <<endl;
      for(int i=0;i<=maxclus;i++) cout << i << "\t" <<maxpoint[i] <<endl;
    }

    // Setting trimming if possible
    char t_opt[2];
    cin.get(comment,150).get();
    cin.get(t_opt,2).get();
    int t_in = strcmp(t_opt,"t");
    int t_of = 0;
    if(t_in == 0 && v_in == 0) {
      cout << "Checking for trimming." << endl;
    }
    if(t_in == 0 && minclus == maxclus && minpoint[1] > 1) {
      t_of = minpoint[1] - 1;
      if(v_in == 0) {
	cout << "Trimming will be applied with an offset of : "<< t_of  << endl;
      }
    }


// SETTING CLUSTERS

    // getting number of clusters
    int numclus = getClusNumber(minpoint,maxpoint,maxclus,nvar);
    
    // clusters array

    CLUSpointArray csarray =new CLUSpoint[numclus+1];

    //initial settings
    for(int i=0;i<=numclus;i++) { 
      csarray[i] = new CLUS[maxclus+1];
      for(int k=0;k<=maxclus;k++) {
	csarray[i][k] = 0 ;
      }
    }

    //first and last array points
    for(int i=0;i<=maxclus;i++) {
	csarray[1][i]=minpoint[i];
	csarray[numclus][i]=maxpoint[i];
    }
   

    for(int i=2;i<numclus;i++) {
      int * cst= new int[maxclus+1];
      for(int k=0;k<=maxclus;k++) cst[k]=csarray[i-1][k];
      int* tt = getNextCluster2(cst,nvar,maxclus);
      for(int k=0;k<=maxclus;k++) csarray[i][k] = tt[k];
      delete[] cst;
    }

    if(v_in == 0) {
      cout << "Total number of clusters: "<< numclus << endl;
    }

// INITIAL SETTINGS
    cout<<setprecision(NUMBER_PREC);   // setting printout's decimal precision


// DATA ARRAY
    TORSpointArray ddat = new TORSpoint[nrec];

    // DATA EXTRACTION
    ifstream indata(dfile);
    int i_line = 0;        
    if(indata.is_open()) {
      string t2line;
      string tline;
      while(i_line < nrec  + f_point - 1) {   
	getline (indata,tline);
	if(i_line >= f_point -1) {
	  char* pch;
	  char* ts;
	  t2line = tline;
	  ts = &tline[0];
	  int idx = 0;
	  pch = strtok(ts," \t");
	  int kk_line = i_line - f_point + 1;
	  ddat[kk_line] =  new TORS[nvar-t_of];
	  while (pch != NULL) {
	    if(idx >= t_of) {
	      ddat[kk_line][idx-t_of] =  atof(pch);
	    }
	    pch = strtok(NULL," \t");
	    idx++;
	  }
	  if(idx != nvar) {
	    cout << "Set and actual number of variables are different!!"<<endl;
	    cout << "Line: "<<i_line<< "\t  Record# : " <<kk_line << endl;
	    cout << "\tInitial: "<<nvar<<"\t\tActual: "<< idx <<endl;
	    cout << "It could be something wrong with data or input parameters."<<endl; 
	    exit(0);
	  }
	} // end of if(i_line >= f_point)
	i_line++;
      } // end of  while(i_line < nrec  + f_point - 1)
    } else {
      cout << "Unable to open datafile: "<< dfile << endl;
      exit(0);
    }
    indata.close();
// end of data reading

    if(v_in == 0) {
      cout << "Data Reading Done. " << i_line << " records extracted" << endl;
    }

    // setting the number of cpus used in calculations
# ifdef _OPENMP
    if(n_proc != 0) {
      omp_set_num_threads(n_proc);
    }
# endif


//Main loop

    double bf_sum =0;

#pragma omp barrier

#pragma omp parallel for shared(ddat,csarray,bf_sum)  schedule(dynamic) //ordered
    for(int ix=1;ix<=numclus;ix++) {
      
      int csize = csarray[ix][0];     //current number of columns
      int * csarr = new int[csize];   //array with selected columns
      int ct = 0;
      for(int k= maxclus - csarray[ix][0] + 1;k<=maxclus;k++) {
	csarr[ct]=csarray[ix][k];
	ct++;
      }
      ANNpointArray pa = new ANNpoint[nrec];  // 

     // copying data 
      for(int jz=0; jz<nrec; jz++) {
	pa[jz] = new ANNcoord[csize];
	for(int kz=0; kz<csize; kz++) {
	  int idxk = csarr[kz] - t_of;                // t_of - offsetting parameter
	  pa[jz][kz] = (ANNcoord) ddat[jz][idxk-1];
	}
      } 
     
      ANNkd_tree* kdTree = new ANNkd_tree(pa,nrec,csize);

      double * dArray = new double[k_nn];
      for(int i = 0; i < k_nn; i++){
	dArray[i] = 0.0;
      }
      
      ANNidxArray indexes = new ANNidx[k_nn];
      ANNdistArray dists = new ANNdist[k_nn];

      //#pragma omp critical
      for(int i = 0; i < nrec; i++){
	kdTree->annkSearch(pa[i], k_nn, indexes, dists, epsilon);

	updateDistEuc(dArray, dists, k_nn);
      }

      estimateEntropyEuc(dArray, csize, nrec, k_nn);

#pragma omp critical //ordered 
      {
 
	if(v_in == 0) {
	  int bf_v = csize*nrec;
	  bf_sum += bf_v*1.0;
	  cout << "Cluster# "<< ix << endl <<" BF: "<< bf_v << "\t BF_TOT: "<< bf_sum << endl;
	}
	cout <<"Selected Dimensions: ";
	int ctt = 0;
	for(int k= maxclus - csarray[ix][0] + 1;k<=maxclus;k++) {
	  csarr[ctt]=csarray[ix][k];
	  ctt++;
	}
	if(csize>1) {
	  for(int k=0;k<csize-1;k++) {
	    cout << csarr[k] <<",";
	  }
	}
	cout << csarr[csize-1];
	cout << endl<<"H:"<<endl;
	for(int k=1;k<=k_nn;k++) {
	  cout << dArray[k-1]<<endl;
	}
	cout << endl<<endl;;
      }

      // memory release
      for(int jz=0; jz<nrec; jz++) {
      	delete pa[jz];
      }
      delete pa;
      delete kdTree;
      delete[] csarr;
      delete[] dArray;
      delete indexes;
      delete dists;
	
    }  //end of main loop

    // memory release

    for(int jz=0; jz<numclus+1; jz++) {
      delete csarray[jz];
    }    
    delete[] csarray;
    for(int jz=0; jz<nrec; jz++) {
      delete ddat[jz];
    }    
    delete[] ddat;    // removing data from memory
    delete[] minpoint;
    delete[] maxpoint;

    if(v_in == 0) {
      cout << "All " << numclus << " clusters done." << endl;
      cout << "Number of total points visited: "<< bf_sum << endl << endl ;
    }

    return 0;
}   // end of program

//==========================================================================================//




// getting the number of expected clusters
int getClusNumber(int* minpoint,int* maxpoint,int maxclus,int nvar){

  int numcs = 1;
  int* fpoint = new int[maxclus+1];
   
  for(int i=0;i<=maxclus;i++) fpoint[i]=minpoint[i];
  
  while(isTheLast(fpoint,maxpoint,fpoint[0],maxclus) == 0) {
    fpoint = getNextCluster2(fpoint,nvar,maxclus);
    numcs++;
  }
  delete[] fpoint;
  return numcs;  // returns number of clusters
}

//-------------------------------------------------------------------

// check whether this is the final cluster
int isTheLast(int* apoint,int* lpoint,int asize,int maxclus) {
    int rvalue = 1;
    if(asize != maxclus) {
	rvalue = 0;
    } else {
      for(int i=1;i<=maxclus;i++) {
	if(apoint[i] != lpoint[i]) rvalue=0;
      }
    }
    return rvalue;
}

//--------------------------------------------------------------------

// gets next cluster
int* getNextCluster2(int* cpoint, int nvar,int maxclus) {

  cpoint[maxclus]++;
  if(cpoint[maxclus]>nvar) {
    cpoint = shiftNextCluster2(cpoint,nvar,maxclus,maxclus);
  }
  return cpoint;
}

//--------------------------------------------------------------------

// increase cluster by one
int* shiftNextCluster2(int* cpoint, int nvar,int maxclus,int posit) {
  int* tpoint = new int[maxclus+1];   
  int av = 0;
  for(int i = 0;i<= maxclus;i++) {
    av = cpoint[i];
    tpoint[i]=av;
  }

  cpoint[posit-1]++;

  if(posit-1 == 0) cout << "ERROR" << endl;

  if(cpoint[posit-1] < nvar-posit+maxclus) { 
    for(int i=posit;i<=maxclus;i++) {
      cpoint[i]=cpoint[posit-1]+i-posit+1;
    }


    if(cpoint[maxclus] > nvar) {
      for(int i = 0;i<= maxclus;i++) {
	av = tpoint[i];
	cpoint[i]=av;
      }

      cpoint = shiftNextCluster2(cpoint,nvar,maxclus,posit-1);

    }
  

  } else {
    for(int i = 0;i<= maxclus;i++) {
      av = tpoint[i];
      cpoint[i]=av;
    }
    cpoint = shiftNextCluster2(cpoint,nvar,maxclus,posit-1);
  }

  int counter = 0;
  for(int i=1;i<=maxclus;i++) {
    if(cpoint[i]>0) counter++;
  }
  if(counter!=cpoint[0]) cpoint[0]=counter;
  delete[] tpoint;
  return cpoint;
}


