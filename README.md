This is a convolution implementation with on the fly Toeplitz matrix generation.
It's a derivative work of OpenBLAS and requires the full OpenBLAS, as only few OpenBLAS
routines have been modified, the rest (gemm kernels) is taken from OpenBLAS itself. It
has been tested on various flavours of Linux. The purpose of this library is to optimize
the Torch spatial convolution module by drastically lowering its memory consumption. As
a side effect, speed is also increased.

## Requirements

OpenBLAS installed in /opt/OpenBLAS

## Use

In Torch require 'openblas-conv'. All the networks will use the modified SpatialConvolutionMM
module. You can enable or disable this new module with openblasconv(true) or openblasconv(false).
It will be active by default when loading the library.

There is also an option to enable this modification partially: openblasconv(true, true) uses the
new updateOutput and accGradParameters, but keeps updateGradInput, as its standard version is
faster. So use this option, when you want to increase speed but you are not interested in
decreasing memory consumption. In any case, the new updateGradInput will be used only for dW and
dH = 1, as it's too demanding to implement the new version for the general case. There is a test.lua
example that tests the correctness and speed of the new algorithms.

This library can be also used stand-alone (without Torch), check the test.c example.
