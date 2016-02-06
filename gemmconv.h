/* Perform c = alpha * a (x) b + beta * c, where

(x) means 2D convolution

a is a 3D tensor (inplanes * inheight * inwidth)
b is the 4D kernel (outplanes * inplanes * kheight * kwidth)
c is a 3D tensor (outplanes * outheight * outwidth)

transa and transb must be 0
m must be outheight * outwidth
n must be outplanes
k must be inplanes * kheight * kwidth
lda is the plane stride for a (inheight * inwidth for contiguous data)
ldb is the outplane stride for b (inplanes * kheight * kwidth for contiguous data)
ldc is the plane stride for c (outheight * outwidth for contiguous data)
os0 is the plane stride for c (outheight * outwidth for contiguous data)
os1 is the line stride for c (outwidth for contiguous data)
ks0 is the inplane stride for b (kheight * kwidth for contiguous data)
ks1 is the the line stride for b (kwidth for continuous data)
is0 is the plane stride for a (inheight * inwidth for contiguous data)
is1 is the plane line stride for a (inwidth for contiguous data)
dW and dH are the steps for the convolution for the width and height dimensions
padW and padH are the additional zeros added per width and height to the input planes
*/
struct sgemmargs {
	long m;
	long n;
	long k;
	long ldb, ldc;
	float alpha, beta;
	float *a, *b, *c;
	long os0, os1, ks0, ks1, is0, is1, ih;
	long dW, dH, padW, padH;
};
void sgemmargs(struct sgemmargs *args);

// Simpler version that creates all the parameters from input friendly data
struct sgemmconv_params {
	long kW, kH;	// Kernel size
	long dW, dH, padW, padH;	// Steps and padding
	long isize[3], osize[3];	// Sizes of the three dimensions
	long istride[2], ostride[2];	// Plane and line strides
	float alpha, beta;
	float *i, *o, *k; // input (a), output (c), kernel (b)
};
int sgemmconv(struct sgemmconv_params *p);

// Double versions
struct dgemmargs {
	long m;
	long n;
	long k;
	long ldb, ldc;
	double alpha, beta;
	double *a, *b, *c;
	long os0, os1, ks0, ks1, is0, is1, ih;
	long dW, dH, padW, padH;
};
void dgemmargs(struct dgemmargs *args);

struct dgemmconv_params {
	long kW, kH;	// Kernel size
	long dW, dH, padW, padH;	// Steps and padding
	long isize[3], osize[3];	// Sizes of the three dimensions
	long istride[2], ostride[2];	// Plane and line strides
	double alpha, beta;
	double *i, *o, *k; // input (a), output (c), kernel (b)
};
int dgemmconv(struct dgemmconv_params *p);
