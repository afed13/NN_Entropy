CPP = g++
#CFLAG = -fopenmp  -g -pg -Wall -fprofile-arcs -ftest-coverage
#CFLAG = -fopenmp -g -Wall -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings 
CFLAG = -fopenmp -O3 -msse4 -mfpmath=sse -Wall -Wextra -Wconversion -Wshadow -Wcast-qual -Wwrite-strings -ansi -pedantic
#CFLAG = -fopenmp -O3 -Wall -Wextra
INCLUDEDIR = -I./ANN 
LIBDIR =  -L/usr/lib 
LIB = -lANN_EUC_v4

.SUFFIXES : .o .cpp 
.cpp.o :
	$(CPP) $(CFLAG) -c $< $(INCLUDEDIR)


nn_entropy_v4.exe : nn_entropy.o common.o
	$(CPP) $(CFLAG) -o $@ nn_entropy.o common.o $(INCLUDEDIR) $(LIBDIR) $(LIB)
	rm -f *.o


clean:
	-rm -f *.o core

realclean: clean
