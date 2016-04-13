This is a convolution implementation with on the fly Toeplitz matrix generation.
It's a derivative work of OpenBLAS and requires the full OpenBLAS, as only few OpenBLAS
routines have been modified, the rest (gemm kernels) is taken from OpenBLAS itself. It
has been tested on various flavours of Linux.

## Building

    make OPENBLASDIR=<OpenBLAS base installation directory>

## Installing

	make install OPENBLASDIR=<OpenBLAS base installation directory>
	
If no OPENBLASDIR is given, /opt/OpenBLAS is assumed.

The library will be installed by default in /usr/local (include and lib).

The purpose of this library is to optimize the Torch spatial convolution module.

An alternative SpatialConvolutionMM.c module of Torch is provided and some test routines.

