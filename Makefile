
CFLAGS=-Wall -O2 
CXXFLAGS=-Wall -O2 -std=c++11

all: nbmcmc

.PHONY: polylog

nbmcmc: main.o ibd.o polylog
	$(CXX) -o nbmcmc main.o ibd.o polylog/libpolylog.a -lm -lgsl -lgslcblas

main.o: main.cpp
	$(CXX) -c $(CXXFLAGS) main.cpp

ibd.o: ibd.cpp ibd.h
	$(CXX) -c $(CXXFLAGS) ibd.cpp

polylog:
	$(MAKE) -C polylog libpolylog.a

clean:
	rm *.o
	$(MAKE) -C polylog clean
	
