OPENBLASDIR = /opt/OpenBLAS
INSTALLDIR = /usr/local
CFLAGS = -Wall -c -fopenmp -fPIC -O3
LIBOBJS = sgemmconv.o dgemmconv.o
CC = gcc
VPATH = generic

UNAME_M := $(shell uname -m)
ifneq ($(LUA_LIBDIR),)
	LIBOBJS += init.o SpatialConvolutionMM.o
	THNN = -lTHNN
	TORCHLIBS = -L$(TORCH)/lib/lua/5.1 -lTHNN
	INSTALLDIR = $(TORCH)/lib/lua/5.1
endif

ifneq ($(filter arm%,$(UNAME_M)),)
	CFLAGS += -march=armv7-a -mfpu=neon
endif

.PHONY : all
all : libopenblas-conv.so

config.h:
	sed 's/OPENBLAS_//' $(OPENBLASDIR)/include/openblas_config.h >config.h

icopy_nopad.h: copy.h.in
	sed 's/_type/_a_nopad/' copy.h.in >icopy_nopad.h

icopy_pad.h: copy.h.in
	sed 's/_type/_a_pad/' copy.h.in >icopy_pad.h

icopy_nopad_t.h: copy.h.in
	sed 's/_type/_a_nopad_t/' copy.h.in >icopy_nopad_t.h

icopy_pad_t.h: copy.h.in
	sed 's/_type/_a_pad_t/' copy.h.in >icopy_pad_t.h

ocopy_conv.h: copy.h.in
	sed 's/_type/_b_conv/' copy.h.in >ocopy_conv.h

sgemmconv.o: gemmconv.c gemmconv.h config.h arch.h param.h icopy_pad.h icopy_nopad.h icopy_pad_t.h icopy_nopad_t.h ocopy_conv.h
	$(CC) $(CFLAGS) gemmconv.c -o sgemmconv.o

dgemmconv.o: gemmconv.c gemmconv.h config.h arch.h param.h icopy_pad.h icopy_nopad.h icopy_pad_t.h icopy_nopad_t.h ocopy_conv.h
	$(CC) $(CFLAGS) -DDODOUBLE gemmconv.c -o dgemmconv.o

init.o: init.c
	$(CC) $(CFLAGS) -I$(LUA_INCDIR)/TH -I. init.c -o init.o

stest.o: test.c sgemmconv.o gemmconv.h
	$(CC) $(CFLAGS) test.c -o stest.o

dtest.o: test.c dgemmconv.o gemmconv.h
	$(CC) $(CFLAGS) -DDODOUBLE test.c -o dtest.o

libopenblas-conv.so: $(LIBOBJS)
	$(CC) -o $@ $(LIBOBJS) -shared -fopenmp -L$(LUA_LIBDIR)/lua/5.1 $(THNN)
	
stest: stest.o sgemmconv.o gemmconv.h
	$(CC) -o $@ stest.o -fopenmp sgemmconv.o -L$(OPENBLASDIR)/lib -lopenblas

dtest: dtest.o dgemmconv.o gemmconv.h
	$(CC) -o $@ dtest.o -fopenmp sgemmconv.o dgemmconv.o -L$(OPENBLASDIR)/lib -lopenblas

test: stest dtest
	./stest
	./dtest

.PHONY : clean

clean:
	rm -f *.o config.h icopy_*.h ocopy_conv.h libopenblas-conv.so stest dtest

install:
	cp libopenblas-conv.so $(INST_LIBDIR)
	cp openblas-conv.lua $(INST_LUADIR)
