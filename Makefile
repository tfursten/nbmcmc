
CFLAGS=-Wall -O2 
CXXFLAGS=-Wall -O2 -std=c++11

all: nbmcmc

.PHONY: polylog

nbmcmc: main.o polylog
	$(CXX) -o nbmcmc main.o polylog/libpolylog.a -lm -lgsl -lgslcblas

main.o: main.cpp
	$(CXX) -c $(CXXFLAGS) main.cpp

polylog:
	$(MAKE) -C polylog libpolylog.a

clean:
	rm *.o
	$(MAKE) -C polylog clean
	
