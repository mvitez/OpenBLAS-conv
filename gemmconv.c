/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

/* Modified by Marko Vitez for Purdue University                     */

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sched.h>
#include "gemmconv.h"
#include "arch.h"
#include "param.h"	// From OpenBLAS, here we only take xGEMM_DEFAULT_UNROLL_x

void *blas_memory_alloc(int);
void blas_memory_free(void *);

#ifndef SWITCH_RATIO
#define SWITCH_RATIO 2
#endif

#ifdef DODOUBLE

typedef double FLOAT;
extern long dgemm_p, dgemm_q, dgemm_r;
int dgemm_oncopy(long m, long n, FLOAT *a, long lda, FLOAT *b);
int dgemm_kernel(long m, long n, long k, FLOAT alpha, FLOAT *sa, FLOAT *sb, FLOAT *c, long ldc);
int dgemm_beta(long m, long n, long dummy1, FLOAT beta, FLOAT *dummy2, long dummy3, FLOAT *dummy4, long dummy5, FLOAT *c, long ldc);
#define fgemmargs dgemmargs
#define fgemmconv_params dgemmconv_params
#define fgemmconv dgemmconv
#define fgemmconv_init dgemmconv_init
#define fgemmconv_exit dgemmconv_exit
#define fgemm_oncopy dgemm_oncopy
#define fgemm_kernel dgemm_kernel
#define fgemm_beta dgemm_beta
#define GEMM_P dgemm_p
#define GEMM_Q dgemm_q
#define GEMM_R dgemm_r
#define GEMM_UNROLL_M DGEMM_DEFAULT_UNROLL_M
#define GEMM_UNROLL_N DGEMM_DEFAULT_UNROLL_N

#else

typedef float FLOAT;
extern long sgemm_p, sgemm_q, sgemm_r;
int sgemm_oncopy(long m, long n, FLOAT *a, long lda, FLOAT *b);
int sgemm_kernel(long m, long n, long k, FLOAT alpha, FLOAT *sa, FLOAT *sb, FLOAT *c, long ldc);
int sgemm_beta(long m, long n, long dummy1, FLOAT beta, FLOAT *dummy2, long dummy3, FLOAT *dummy4, long dummy5, FLOAT *c, long ldc);
#define fgemmargs sgemmargs
#define fgemmconv_params sgemmconv_params
#define fgemmconv sgemmconv
#define fgemmconv_init sgemmconv_init
#define fgemmconv_exit sgemmconv_exit
#define fgemm_oncopy sgemm_oncopy
#define fgemm_kernel sgemm_kernel
#define fgemm_beta sgemm_beta
#define GEMM_P sgemm_p
#define GEMM_Q sgemm_q
#define GEMM_R sgemm_r
#define GEMM_UNROLL_M SGEMM_DEFAULT_UNROLL_M
#define GEMM_UNROLL_N SGEMM_DEFAULT_UNROLL_N

#endif

#define MAX_CPU_NUMBER 32
#define CACHE_LINE_SIZE 8
#define DIVIDE_RATE 2
#define ZERO 0

#define BUFFER_SIZE ((GEMM_P * GEMM_Q * sizeof(FLOAT) + GEMM_ALIGN) & ~GEMM_ALIGN) * 3
#define GEMM_ALIGN 0x03fffUL

#define MIN(a,b) ((a)>(b) ? (b) : (a))

typedef struct {
	volatile long working[MAX_CPU_NUMBER][CACHE_LINE_SIZE * DIVIDE_RATE];
} job_t;

#define OCOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) fgemm_oncopy(M, N, (FLOAT *)(A) + ((X) + (Y) * (LDA)), LDA, BUFFER);
#define KERNEL_OPERATION(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) fgemm_kernel(M, N, K, ALPHA, SA, SB, (FLOAT *)(C) + ((X) + (Y) * LDC), LDC)
#define BETA_OPERATION(M_FROM, M_TO, N_FROM, N_TO, BETA, C, LDC) fgemm_beta((M_TO) - (M_FROM), (N_TO - N_FROM), 0, \
		  BETA, NULL, 0, NULL, 0, (FLOAT *)(C) + (M_FROM) + (N_FROM) * (LDC), LDC)

static struct {
	int x1, y1, plane;
} *xtab;

static struct {
	int x1, y1;
} *ytab;

static void calctabs(struct fgemmargs *args)
{
	int x, y, xmax, ymax;

	xmax = args->transa ? args->m : args->k;
	ymax = args->transa ? args->k : args->m;
	xtab = malloc(sizeof(*xtab) * xmax);
	ytab = malloc(sizeof(*ytab) * ymax);
	for(x = 0; x < xmax; x++)
	{
		xtab[x].plane = x / args->kHW;
		int x1 = x % args->kHW;
		int y1 = x1 / args->kW - args->padH;
		x1 = x1 % args->kW - args->padW;
		xtab[x].y1 = y1 * args->is1;
		xtab[x].x1 = x1;
	}
	for(y = 0; y < ymax; y++)
	{
		int y1 = y / args->ow * args->dH;
		int x1 = y % args->ow * args->dW;
		ytab[y].y1 = y1 * args->is1;
		ytab[y].x1 = x1;
	}
}

static FLOAT get_a_pad(struct fgemmargs *args, int x, int y)
{
	int x1 = xtab[x].x1 + ytab[y].x1;
	int y1 = xtab[x].y1 + ytab[y].y1;
	if(y1 < 0 || y1 >= args->is0 || x1 < 0 || x1 >= args->is1)
		return 0;
	return args->a[xtab[x].plane*args->is0 + y1 + x1];
}

static FLOAT get_a_nopad(struct fgemmargs *args, int x, int y)
{
	int x1 = xtab[x].x1 + ytab[y].x1;
	int y1 = xtab[x].y1 + ytab[y].y1;
	return args->a[xtab[x].plane*args->is0 + y1 + x1];
}

static FLOAT get_a_pad_t(struct fgemmargs *args, int y, int x)
{
	int x1 = xtab[x].x1 + ytab[y].x1;
	int y1 = xtab[x].y1 + ytab[y].y1;
	if(y1 < 0 || y1 >= args->is0 || x1 < 0 || x1 >= args->is1)
		return 0;
	return args->a[xtab[x].plane*args->is0 + y1 + x1];
}

static FLOAT get_a_nopad_t(struct fgemmargs *args, int y, int x)
{
	int x1 = xtab[x].x1 + ytab[y].x1;
	int y1 = xtab[x].y1 + ytab[y].y1;
	return args->a[xtab[x].plane*args->is0 + y1 + x1];
}

static FLOAT get_b_conv(struct fgemmargs *args, int x, int y)
{
	int plane = x / args->kHW;
	int rest = x % args->kHW;
	int iW = rest % args->kW;
	int iH = rest / args->kW;
	FLOAT a = args->b[plane * args->kPHW + y * args->kW * args->kH +
		(args->kH - iH - 1) * args->kW + (args->kW - iW - 1)];
	return a;
}

#define GEMM_UNROLL_a_pad GEMM_UNROLL_M
#define GEMM_UNROLL_a_nopad GEMM_UNROLL_M
#define GEMM_UNROLL_a_pad_t GEMM_UNROLL_M
#define GEMM_UNROLL_a_nopad_t GEMM_UNROLL_M
#define GEMM_UNROLL_b_conv GEMM_UNROLL_N

#include "icopy_pad.h"
#include "icopy_nopad.h"
#include "icopy_pad_t.h"
#include "icopy_nopad_t.h"
#include "ocopy_conv.h"

static void icopy_operation(int m, int n, struct fgemmargs *args, int x, int y, FLOAT *b)
{
	if(args->transa)
	{
		if(args->padW || args->padH)
			copy_operation_a_pad_t(m, n, args, x, y, b);
		else copy_operation_a_nopad_t(m, n, args, x, y, b);
	} else {
		if(args->padW || args->padH)
			copy_operation_a_pad(m, n, args, x, y, b);
		else copy_operation_a_nopad(m, n, args, x, y, b);
	}
}

static job_t job[MAX_CPU_NUMBER];

static int gemm_thread(long mypos, long nthreads, struct fgemmargs *args, long *range_m, long *range_n, FLOAT *sa, FLOAT *sb)
{
	FLOAT *buffer[DIVIDE_RATE];
	long m_from, m_to, n_from, n_to;
	long xxx, bufferside;
	long ls, min_l, jjs, min_jj;
	long is, min_i, div_n;
	long i, current;
	long l1stride;
	long m = args->m;
	long n = args->n;
	long k = args->k;
	FLOAT alpha = args->alpha;
	FLOAT beta = args->beta;
	FLOAT *b = args->b;
	FLOAT *c = args->c;
	long ldb = args->ldb;
	long ldc = args->ldc;

	m_from = 0;
	m_to = m;

	if(range_m)
	{
		m_from = range_m[mypos + 0];
		m_to   = range_m[mypos + 1];
	}
	n_from = 0;
	n_to   = n;
	if (range_n)
	{
		n_from = range_n[mypos + 0];
		n_to   = range_n[mypos + 1];
	}
	if(beta != 1)
		BETA_OPERATION(m_from, m_to, 0, n, beta, c, ldc);
	if(k == 0 || alpha == 0)
		return 0;
	div_n = (n_to - n_from + DIVIDE_RATE - 1) / DIVIDE_RATE;
	buffer[0] = sb;
	for (i = 1; i < DIVIDE_RATE; i++)
		buffer[i] = buffer[i - 1] + GEMM_Q * ((div_n + GEMM_UNROLL_N - 1) & ~(GEMM_UNROLL_N - 1));
	for(ls = 0; ls < k; ls += min_l)
	{
		min_l = k - ls;
		if (min_l >= GEMM_Q * 2)
			min_l  = GEMM_Q;
		else if (min_l > GEMM_Q)
			min_l = (min_l + 1) / 2;
		l1stride = 1;
		min_i = m_to - m_from;
		if (min_i >= GEMM_P * 2)
			min_i = GEMM_P;
		else if(min_i > GEMM_P)
			min_i = (min_i / 2 + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1);
		else if (nthreads == 1)
			l1stride = 0;
		//printf("icopy%ld (%ld,%ld)%ld (%ld,%ld)\n", mypos, min_l, min_i, lda, ls, m_from);
		icopy_operation(min_l, min_i, args, ls, m_from, sa);
		div_n = (n_to - n_from + DIVIDE_RATE - 1) / DIVIDE_RATE;
		for (xxx = n_from, bufferside = 0; xxx < n_to; xxx += div_n, bufferside ++)
		{
			/* Make sure if no one is using buffer */
			for (i = 0; i < nthreads; i++)
				while (job[mypos].working[i][CACHE_LINE_SIZE * bufferside])
					{YIELDING;}
			for(jjs = xxx; jjs < MIN(n_to, xxx + div_n); jjs += min_jj)
			{
				min_jj = MIN(n_to, xxx + div_n) - jjs;
				if(min_jj >= 3*GEMM_UNROLL_N)
					min_jj = 3*GEMM_UNROLL_N;
				else if (min_jj > GEMM_UNROLL_N)
					min_jj = GEMM_UNROLL_N;
				if(args->conv)
					copy_operation_b_conv(min_l, min_jj, args, ls, jjs, buffer[bufferside] + min_l * (jjs - xxx) * l1stride);
				else OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs, buffer[bufferside] + min_l * (jjs - xxx) * l1stride);
				KERNEL_OPERATION(min_i, min_jj, min_l, alpha, sa, buffer[bufferside] + min_l * (jjs - xxx) * l1stride, c, ldc, m_from, jjs);
			}
			for (i = 0; i < nthreads; i++)
				job[mypos].working[i][CACHE_LINE_SIZE * bufferside] = (long)buffer[bufferside];
			WMB;
		}
		current = mypos;
		do {
			current ++;
			if(current >= nthreads)
				current = 0;
			div_n = (range_n[current + 1]  - range_n[current] + DIVIDE_RATE - 1) / DIVIDE_RATE;
			for (xxx = range_n[current], bufferside = 0; xxx < range_n[current + 1]; xxx += div_n, bufferside ++)
			{
				if (current != mypos)
				{
					/* thread has to wait */
					while(job[current].working[mypos][CACHE_LINE_SIZE * bufferside] == 0)
						{YIELDING;}
					KERNEL_OPERATION(min_i, MIN(range_n[current + 1]  - xxx,  div_n), min_l, alpha, sa,
						(FLOAT *)job[current].working[mypos][CACHE_LINE_SIZE * bufferside], c, ldc, m_from, xxx);
				}
				if (m_to - m_from == min_i)
					job[current].working[mypos][CACHE_LINE_SIZE * bufferside] &= 0;
			}
		} while (current != mypos);
		for(is = m_from + min_i; is < m_to; is += min_i)
		{
			min_i = m_to - is;
			if (min_i >= GEMM_P * 2)
				min_i = GEMM_P;
			else if (min_i > GEMM_P)
				min_i = ((min_i + 1) / 2 + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1);
			//printf("icopya%ld (%ld,%ld)%ld (%ld,%ld)\n", mypos, min_l, min_i, lda, ls, is);
			icopy_operation(min_l, min_i, args, ls, is, sa);
			current = mypos;
			do {
				div_n = (range_n[current + 1] - range_n[current] + DIVIDE_RATE - 1) / DIVIDE_RATE;
				for (xxx = range_n[current], bufferside = 0; xxx < range_n[current + 1]; xxx += div_n, bufferside ++)
				{
					KERNEL_OPERATION(min_i, MIN(range_n[current + 1] - xxx, div_n), min_l, alpha, sa,
						(FLOAT *)job[current].working[mypos][CACHE_LINE_SIZE * bufferside], c, ldc, is, xxx);
					if(is + min_i >= m_to)
					{
						/* Thread doesn't need this buffer any more */
						job[current].working[mypos][CACHE_LINE_SIZE * bufferside] &= 0;
						WMB;
					}
				}
				current ++;
				if(current >= nthreads)
					current = 0;
			} while (current != mypos);
		}
	}
	for (i = 0; i < nthreads; i++)
		for (xxx = 0; xxx < DIVIDE_RATE; xxx++)
			while (job[mypos].working[i][CACHE_LINE_SIZE * xxx] )
				{YIELDING;}
	return 0;
}

static int gemm_single(int mypos, struct fgemmargs *args, FLOAT *sa, FLOAT *sb)
{
	long m_from, m_to, n_from, n_to;

	long ls, is, js;
	long min_l, min_i, min_j;
	long jjs, min_jj;
	long l1stride, gemm_p, l2size;
	long m = args->m;
	long n = args->n;
	long k = args->k;
	FLOAT alpha = args->alpha;
	FLOAT beta = args->beta;
	FLOAT *b = args->b;
	FLOAT *c = args->c;
	long ldb = args->ldb;
	long ldc = args->ldc;


	m_from = 0;
	m_to   = m;
	n_from = 0;
	n_to   = n;
	if (beta != 1)
		BETA_OPERATION(m_from, m_to, n_from, n_to, beta, c, ldc);

	if((k == 0) || (alpha == 0))
		return 0;
	l2size = GEMM_P * GEMM_Q;

	for(js = n_from; js < n_to; js += GEMM_R)
	{
		min_j = n_to - js;
		if (min_j > GEMM_R)
			min_j = GEMM_R;

		for(ls = 0; ls < k; ls += min_l)
		{
			min_l = k - ls;
			if(min_l >= GEMM_Q * 2)
			{
				gemm_p = GEMM_P;
				min_l  = GEMM_Q;
			} else {
				if(min_l > GEMM_Q)
					min_l = (min_l / 2 + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1);
				gemm_p = ((l2size / min_l + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1));
				while (gemm_p * min_l > l2size)
					gemm_p -= GEMM_UNROLL_M;
			}
			/* First, we have to move data A to L2 cache */
			min_i = m_to - m_from;
			l1stride = 1;
			if(min_i >= GEMM_P * 2)
				min_i = GEMM_P;
			else if(min_i > GEMM_P)
				min_i = (min_i / 2 + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1);
			else l1stride = 0;
			icopy_operation(min_l, min_i, args, ls, m_from, sa);
			for(jjs = js; jjs < js + min_j; jjs += min_jj)
			{
				min_jj = min_j + js - jjs;
				if(min_jj >= 3*GEMM_UNROLL_N)
					min_jj = 3*GEMM_UNROLL_N;
				else if(min_jj > GEMM_UNROLL_N)
					min_jj = GEMM_UNROLL_N;
				if(args->conv)
					copy_operation_b_conv(min_l, min_jj, args, ls, jjs, sb + min_l * (jjs - js) * l1stride);
				else OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs, sb + min_l * (jjs - js) * l1stride);
				KERNEL_OPERATION(min_i, min_jj, min_l, alpha, sa,
					sb + min_l * (jjs - js) * l1stride, c, ldc, m_from, jjs);
			}

			for(is = m_from + min_i; is < m_to; is += min_i)
			{
				min_i = m_to - is;
				if(min_i >= GEMM_P * 2)
					min_i = GEMM_P;
				else if(min_i > GEMM_P)
					min_i = (min_i / 2 + GEMM_UNROLL_M - 1) & ~(GEMM_UNROLL_M - 1);
				icopy_operation(min_l, min_i, args, ls, is, sa);
				KERNEL_OPERATION(min_i, min_j, min_l, alpha, sa, sb, c, ldc, is, js);
			} /* end of is */
		} /* end of js */
	} /* end of ls */

	return 0;
}

void fgemmargs(struct fgemmargs *args)
{
	long range_M[MAX_CPU_NUMBER + 1];
	long range_N[MAX_CPU_NUMBER + 1];

	long num_cpu_m, num_cpu_n;

	long width, i, j, k1, js;
	long m1, n1, n_from, n_to;

#if 0
	printf("conv=%d, transa=%d, alpha=%f, beta=%f\n", args->transa, args->conv, args->alpha, args->beta);
	printf("m=%ld, n=%ld, k=%ld, ldb=%ld, ldc=%ld\n", args->m, args->n, args->k, args->ldb, args->ldc);
	printf("ow=%ld, kPHW=%ld, kHW=%ld, kW=%ld, is0=%ld, is1=%ld, ih=%ld\n", args->ow, args->kPHW, args->kHW, args->kW,
		args->is0, args->is1, args->ih);
	printf("dW=%ld, dH=%ld, padW=%ld, padH=%ld\n", args->dW, args->dH, args->padW, args->padH);
#endif
	int threads_num = omp_get_max_threads();
	FLOAT *buffer = (FLOAT *)blas_memory_alloc(0);
	FLOAT *sa = (FLOAT *)((long)buffer +GEMM_DEFAULT_OFFSET_A);
	FLOAT *sb = (FLOAT *)(((long)sa + ((GEMM_P * GEMM_Q * sizeof(FLOAT) + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_DEFAULT_OFFSET_B);
	range_M[0] = 0;
	m1 = args->m;
	num_cpu_m  = 0;

	if(omp_in_parallel())
	{
		if(args->kHW && omp_get_thread_num() == 0)
			calctabs(args);
#pragma omp barrier
		gemm_single(omp_get_thread_num(), args, sa, sb);
#pragma omp barrier
		if(args->kHW && omp_get_thread_num() == 0)
		{
			free(xtab);
			free(ytab);
		}
		blas_memory_free(buffer);
		return;
	}
	if(args->kHW)
		calctabs(args);
	if((args->m < threads_num * SWITCH_RATIO) || (args->n < threads_num * SWITCH_RATIO))
	{
		gemm_single(0, args, sa, sb);
		if(args->kHW)
		{
			free(xtab);
			free(ytab);
		}
		blas_memory_free(buffer);
		return;
	}

	while(m1 > 0)
	{
		width = (m1 + threads_num - num_cpu_m - 1) / (threads_num - num_cpu_m);
		m1 -= width;
		if(m1 < 0)
			width = width + m1;
		range_M[num_cpu_m + 1] = range_M[num_cpu_m] + width;
		num_cpu_m++;
	}
	n_from = 0;
	n_to   = args->n;

	for(js = n_from; js < n_to; js += GEMM_R * threads_num)
	{
		n1 = n_to - js;
		if (n1 > GEMM_R * threads_num)
			n1 = GEMM_R * threads_num;
		range_N[0] = js;
		num_cpu_n  = 0;
		while (n1 > 0)
		{
			width = (n1 + threads_num - num_cpu_n - 1) / (threads_num - num_cpu_n);
			n1 -= width;
			if(n1 < 0)
				width = width + n1;
			range_N[num_cpu_n + 1] = range_N[num_cpu_n] + width;
			num_cpu_n++;
		}
		for (j = 0; j < num_cpu_m; j++)
			for (i = 0; i < num_cpu_m; i++)
				for (k1 = 0; k1 < DIVIDE_RATE; k1++)
					job[j].working[i][CACHE_LINE_SIZE * k1] = 0;
#pragma omp parallel for
		for(i = 0; i < threads_num; i++)
			gemm_thread(i, threads_num, args, range_M, range_N, (FLOAT *)((char *)sa + i * BUFFER_SIZE), (FLOAT *)((char *)sb + i * BUFFER_SIZE));
	}
	if(args->kHW)
	{
		free(xtab);
		free(ytab);
	}
	blas_memory_free(buffer);
}

int fgemmconv(struct fgemmconv_params *p)
{
	struct fgemmargs a;
	// Assure output dimensions are correct
	if(p->osize[1] != (p->isize[1] + 2 * p->padH - p->ksize[2]) / p->dH + 1)
		return -1;
	if(p->osize[2] != (p->isize[2] + 2 * p->padW - p->ksize[3]) / p->dW + 1)
		return -1;
	if(p->transi)
	{
		a.m = p->isize[0] * p->ksize[2] * p->ksize[3];
		a.n = p->osize[0];
		a.k = p->osize[1] * p->osize[2];
		a.ldb = a.k;
		a.ldc = p->ostride0;
	} else {
		a.m = p->osize[1] * p->osize[2];
		a.n = p->osize[0];
		a.k = p->isize[0] * p->ksize[2] * p->ksize[3];
		a.ldb = a.k;
		a.ldc = p->ostride0;
	}
	a.transa = p->transi;
	a.conv = p->conv;
	a.alpha = p->alpha;
	a.beta = p->beta;
	a.a = p->i;
	a.b = p->k;
	a.c = p->o;
	a.kH = p->ksize[2];
	a.kW = p->ksize[3];
	a.kHW = p->ksize[2] * p->ksize[3];
	a.kPHW = p->ksize[1] * p->ksize[2] * p->ksize[3];
	a.is0 = p->istride[0];
	a.is1 = p->istride[1];
	a.ow = p->owidth;
	a.ih =  p->isize[1];
	a.dW = p->dW;
	a.dH = p->dH;
	a.padW = p->padW;
	a.padH = p->padH;
	fgemmargs(&a);
	return 0;
}
