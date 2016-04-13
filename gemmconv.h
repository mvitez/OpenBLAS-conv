#ifndef _GEMMCONV_H_INCLUDED_
#define _GEMMCONV_H_INCLUDED_

/* Perform c = alpha * a (x) b + beta * c, where

(x) means 2D cross-correlation

a is a 3D tensor (inplanes * inheight * inwidth)
b is the 4D kernel (outplanes * inplanes * kheight * kwidth)
c is a 3D tensor (outplanes * outheight * outwidth)

m must be outheight * outwidth
n must be outplanes
k must be inplanes * kheight * kwidth
ldb is the outplane stride for b (inplanes * kheight * kwidth for contiguous data)
ldc is the plane stride for c (outheight * outwidth for contiguous data)
ow is the output width
kW is the kernel width
kH is the kernel height
kHW is kH * kW
kPHW is ninputplanes * kH * kW
is0 is the plane stride for a (inheight * inwidth for contiguous data)
is1 is the line stride for a (inwidth for contiguous data)
dW and dH are the steps for the convolution for the width and height dimensions
padW and padH are the additional zeros added per width and height to the input planes

If transa is 1, the generated Toeplitz matrix will be transposed and b is supposed to be
the output instead of the kernel, which will be calculated instead (reverse cross-correlation)
If conv is 1, the weights will be transposed, the operation will be a full convolution instead
of a valid cross-correlation
*/

struct sgemmargs {
	int transa, conv;
	long m;
	long n;
	long k;
	long ldb, ldc;
	float alpha, beta;
	float *a, *b, *c;
	long ow, kH, kW, kHW, kPHW, is0, is1, ih;
	long dW, dH, padW, padH;
};
void sgemmargs(struct sgemmargs *args);

// Simpler version that creates all the parameters from input friendly data
struct sgemmconv_params {
	int conv;	// 1 = convolution, 0 = cross-correlation
	int transi;	// 1 = transpose input, 0 = don't transpose input
	long dW, dH, padW, padH;	// Steps and padding
	long isize[3], osize[3];	// Sizes of the three dimensions
	long ksize[4];	// Sizes of the kernel
	long istride[2];	// Input plane and line strides
	long ostride0;	// Output plane stride
	long owidth;	// Output width
	float alpha, beta;
	float *i, *o, *k; // input (a), output (c), kernel (b)
};
int sgemmconv(struct sgemmconv_params *p);

// Double versions
struct dgemmargs {
	int transa, conv;
	long m;
	long n;
	long k;
	long ldb, ldc;
	double alpha, beta;
	double *a, *b, *c;
	long ow, kH, kW, kHW, kPHW, is0, is1, ih;
	long dW, dH, padW, padH;
};
void dgemmargs(struct dgemmargs *args);

struct dgemmconv_params {
	int conv;	// 1 = convolution, 0 = cross-correlation
	int transi;	// 1 = transpose input, 0 = don't transpose input
	long dW, dH, padW, padH;	// Steps and padding
	long isize[3], osize[3];	// Sizes of the three dimensions
	long ksize[4];	// Sizes of the kernel
	long istride[2];	// Input plane and line strides
	long ostride0;	// Output plane stride
	long owidth;	// Output width
	double alpha, beta;
	double *i, *o, *k; // input (a), output (c), kernel (b)
};
int dgemmconv(struct dgemmconv_params *p);

#endif