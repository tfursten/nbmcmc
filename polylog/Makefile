
OBJ=polylog.o const.o fac.o mtherr.o polevl.o powi.o spence.o zetac.o

all: libpolylog.a

libpolylog.a: $(OBJ)
	ar rcs libpolylog.a $(OBJ)

polylog.o: polylog.c polylog.h
	$(CC) -c $(CFLAGS) polylog.c polylog.h

const.o: const.c
	$(CC) -c $(CFLAGS) const.c

fac.o: fac.c
	$(CC) -c $(CFLAGS) fac.c

mtherr.o: mtherr.c
	$(CC) -c $(CFLAGS) mtherr.c

polevl.o: polevl.c
	$(CC) -c $(CFLAGS) polevl.c

powi.o: powi.c
	$(CC) -c $(CFLAGS) powi.c

spence.o: spence.c
	$(CC) -c $(CFLAGS) spence.c

zetac.o: zetac.c
	$(CC) -c $(CFLAGS) zetac.c

clean:
	rm *.o
	