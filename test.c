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
	FLOAT sum;
	p.dW = p.dH = 2;
	p.padW = p.padH = 2;
	p.kW = p.kH = 3;
	p.isize[0] = 3;
	p.isize[1] = 720;
	p.isize[2] = 1280;
	p.osize[0] = 2;
	p.osize[1] = (p.isize[1] + 2 * p.padH - p.kH) / p.dH + 1;
	p.osize[2] = (p.isize[2] + 2 * p.padW - p.kW) / p.dW + 1;
	p.istride[0] = p.isize[1] * p.isize[2];
	p.istride[1] = p.isize[2];
	p.ostride[0] = p.osize[1] * p.osize[2];
	p.ostride[1] = p.osize[2];
	p.alpha = p.beta = 1;
	p.i = calloc(p.isize[0] * p.isize[1] * p.isize[2], sizeof(*p.i));
	p.k = calloc(p.isize[0] * p.osize[0] * p.kH * p.kW, sizeof(*p.i));
	p.o = calloc(p.osize[0] * p.osize[1] * p.osize[2], sizeof(*p.i));
	for(i = 0; i < p.isize[0] * p.isize[1] * p.isize[2]; i++)
		p.i[i] = i % 100 * 0.01;
	for(i = 0; i < p.isize[0] * p.osize[0] * p.kW * p.kH; i++)
		p.k[i] = i % 100 * 0.01;
	fgemmconv(&p);
	sum = 0;
	for(i = 0; i < p.osize[0] * p.osize[1] * p.osize[2]; i++)
		sum += p.o[i];
#ifdef DODOUBLE
	if(fabs(sum - 1626531.84) < 1)
		printf("The double output is correct\n");
	else printf("The double output is wrong\n");
#else
	if(fabs(sum - 1626206.125) < 1)
		printf("The float output is correct\n");
	else printf("The float output is wrong\n");
#endif
	return 0;
}
