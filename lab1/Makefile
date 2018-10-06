all: ./lab1  

clean:
	rm ./lab1

# MKL needs module load intel/17.0.1
MKLDIR := /share/apps/intel/17.0.1/mkl
CPPFLAGS := -I$(MKLDIR)/include -D_GNU_SOURCE
CFLAGS := -O2 -W -Wall 

LDFLAGS := -L$(MKLDIR)/lib/intel64
LDLIBS := -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm

lab1:
	$(CC) -o 

.PHONY: clean
