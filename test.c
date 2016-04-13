#include <stdio.h>
#include <string.h>
#include <math.h>
#include "gemmconv.h"

#ifdef DODOUBLE
typedef double FLOAT;
#define fgemmconv dgemmconv
#define fgemmconv_params dgemmconv_params
#else
typedef float FLOAT;
#define fgemmconv sgemmconv
#define fgemmconv_params sgemmconv_params
#endif
int main()
{
	int i;
	struct fgemmconv_params p;
	double sum;
	p.conv = 0;
	p.transi = 0;
	p.dW = p.dH = 2;
	p.padW = p.padH = 2;
	p.ksize[2] = p.ksize[3] = 3;
	p.isize[0] = 3;
	p.isize[1] = 720;
	p.isize[2] = 1280;
	p.osize[0] = 2;
	p.osize[1] = (p.isize[1] + 2 * p.padH - p.ksize[2]) / p.dH + 1;
	p.osize[2] = (p.isize[2] + 2 * p.padW - p.ksize[3]) / p.dW + 1;
	p.istride[0] = p.isize[1] * p.isize[2];
	p.istride[1] = p.isize[2];
	p.ostride0 = p.osize[1] * p.osize[2];
	p.owidth = p.osize[2];
	p.alpha = p.beta = 1;
	p.i = calloc(p.isize[0] * p.isize[1] * p.isize[2], sizeof(*p.i));
	p.k = calloc(p.isize[0] * p.osize[0] * p.ksize[2] * p.ksize[3], sizeof(*p.i));
	p.o = calloc(p.osize[0] * p.osize[1] * p.osize[2], sizeof(*p.i));
	for(i = 0; i < p.isize[0] * p.isize[1] * p.isize[2]; i++)
		p.i[i] = i % 100 * 0.01;
	for(i = 0; i < p.isize[0] * p.osize[0] * p.ksize[3] * p.ksize[2]; i++)
		p.k[i] = i % 100 * 0.01;
	fgemmconv(&p);
	sum = 0;
	for(i = 0; i < p.osize[0] * p.osize[1] * p.osize[2]; i++)
		sum += p.o[i];
	if(fabs(sum - 1626531.0) < 1)
#ifdef DODOUBLE
		printf("The double output is correct\n");
	else printf("The double output is wrong\n");
#else
		printf("The float output is correct\n");
	else printf("The float output is wrong\n");
#endif
	return 0;
}
