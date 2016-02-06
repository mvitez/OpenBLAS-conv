OPENBLASDIR = /opt/OpenBLAS

CFLAGS = -Wall -c -fopenmp -fPIC -O3
CC = gcc

.PHONY : all
all : libopenblas-conv.so stest dtest test

config.h:
	sed 's/OPENBLAS_//' $(OPENBLASDIR)/include/openblas_config.h >config.h

sgemmconv.o: gemmconv.c gemmconv.h config.h arch.h param.h
	$(CC) $(CFLAGS) gemmconv.c -o sgemmconv.o

dgemmconv.o: gemmconv.c gemmconv.h config.h arch.h param.h
	$(CC) $(CFLAGS) -DDODOUBLE gemmconv.c -o dgemmconv.o

stest.o: test.c sgemmconv.o gemmconv.h
	$(CC) $(CFLAGS) test.c -o stest.o

dtest.o: test.c dgemmconv.o gemmconv.h
	$(CC) $(CFLAGS) -DDODOUBLE test.c -o dtest.o

libopenblas-conv.so: sgemmconv.o dgemmconv.o
	$(CC) -o $@ sgemmconv.o dgemmconv.o -shared -fopenmp
	
stest: stest.o sgemmconv.o gemmconv.h
	$(CC) -o $@ stest.o -fopenmp sgemmconv.o -L$(OPENBLASDIR)/lib -lopenblas

dtest: dtest.o dgemmconv.o gemmconv.h
	$(CC) -o $@ dtest.o -fopenmp sgemmconv.o dgemmconv.o -L$(OPENBLASDIR)/lib -lopenblas

test: stest dtest
	./stest
	./dtest

.PHONY : clean
clean :
	rm -f *.o config.h libopenblas-conv.so stest dtest

install:
	cp libopenblas-conv.so $(OPENBLASDIR)/lib
	cp sgemmconv.h $(OPENBLASDIR)/include

uninstall:
	rm $(OPENBLASDIR)/lib/libopenblas-conv.so
	rm $(OPENBLASDIR)/include/sgemmconv.h
