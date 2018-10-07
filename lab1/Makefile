all: ./lab1-c5-c6  ./lab1-c1-c2

clean:
	rm ./lab1-c5-c6 ./lab1-c1-c2 ./lab1-c5-c6.o

CC = gcc

# MKL needs module load intel/17.0.1
MKLDIR := /share/apps/intel/17.0.1/mkl
CPPFLAGS := -I$(MKLDIR)/include -D_GNU_SOURCE
CFLAGS := -O2 -W -Wall -g

LDFLAGS := -L$(MKLDIR)/lib/intel64
LDLIBS := -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm

lab1-c5-c6: lab1-c5-c6.o
	$(CC) $(CFLAGS) $(CPPFLAGS) -o lab1-c5-c6 lab1-c5-c6.o $(LDFLAGS) $(LDLIBS)

lab1-c5-c6.o: lab1-c5-c6.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c lab1-c5-c6.c -o lab1-c5-c6.o 

lab1-c1-c2: lab1-c1-c2.o
	$(CC) -W -Wall -O3 -o lab1-c1-c2 lab1-c1-c2.o

lab1-c1-c2.o: lab1-c1-c2.c
	$(CC) -W -Wall -O3 -c lab1-c1-c2.c -o lab1-c1-c2.o

.PHONY: clean
