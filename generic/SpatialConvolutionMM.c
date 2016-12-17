#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMM.c"
#else

TH_API void THNN_(unfolded_acc)(
          THTensor *finput,
          THTensor *input,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int nInputPlane,
          int inputWidth, int inputHeight,
          int outputWidth, int outputHeight);
TH_API void THNN_(unfolded_copy)(
          THTensor *finput,
          THTensor *input,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int nInputPlane,
          int inputWidth, int inputHeight,
          int outputWidth, int outputHeight);

#ifdef BLASCONV

#include "gemmconv.h"

void THTensor_(convmm)(THTensor *output, real beta, real alpha, THTensor *weight, THTensor *input, int kH, int kW,
                 int dH, int dW, int padH, int padW, int op)
{
#if defined(TH_REAL_IS_DOUBLE)
  struct dgemmconv_params p;
  
  p.conv = op == 1;
  p.transi = op == 2;
  p.ksize[3] = kW;
  p.ksize[2] = kH;
  p.ksize[1] = weight->size[1] / (kH * kW);
  p.ksize[0] = weight->size[0];
  p.dW = dW;
  p.dH = dH;
  p.padW = padW;
  p.padH = padH;
  memcpy(p.isize, input->size, sizeof(p.isize[0]) * 3);
  memcpy(p.istride, input->stride, sizeof(p.istride[0]) * 2);
  if(op == 2)
  {
    // Transpose input
    memcpy(p.osize, weight->size, sizeof(p.osize[0]) * 3);
	p.ostride0 = output->stride[0];
	p.owidth = weight->size[2];
  } else {
    if(op == 1)
	{
		// Full convolution instead of valid convolution
		p.padW = kW-1 - padW;
		p.padH = kH-1 - padH;
	}
    memcpy(p.osize, output->size, sizeof(p.osize[0]) * 3);
	p.ostride0 = output->stride[0];
	p.owidth = output->size[2];
  }
  p.alpha = alpha;
  p.beta = beta;
  p.i = THTensor_(data)(input);
  p.o = THTensor_(data)(output);
  p.k = THTensor_(data)(weight);
  dgemmconv(&p);
#else
  struct sgemmconv_params p;
  
  p.conv = op == 1;
  p.transi = op == 2;
  p.ksize[3] = kW;
  p.ksize[2] = kH;
  p.ksize[1] = weight->size[1] / (kH * kW);
  p.ksize[0] = weight->size[0];
  p.dW = dW;
  p.dH = dH;
  p.padW = padW;
  p.padH = padH;
  memcpy(p.isize, input->size, sizeof(p.isize[0]) * 3);
  memcpy(p.istride, input->stride, sizeof(p.istride[0]) * 2);
  if(op == 2)
  {
    // Transpose input
    memcpy(p.osize, weight->size, sizeof(p.osize[0]) * 3);
	p.ostride0 = output->stride[0];
	p.owidth = weight->size[2];
  } else {
    if(op == 1)
	{
		// Full convolution instead of valid convolution
		p.padW = kW-1 - padW;
		p.padH = kH-1 - padH;
	}
    memcpy(p.osize, output->size, sizeof(p.osize[0]) * 3);
	p.ostride0 = output->stride[0];
	p.owidth = output->size[2];
  }
  p.alpha = alpha;
  p.beta = beta;
  p.i = THTensor_(data)(input);
  p.o = THTensor_(data)(output);
  p.k = THTensor_(data)(weight);
  sgemmconv(&p);
#endif  
}
#endif

static void THNN_(SpatialConvolutionMM_updateOutput_frame)(THTensor *input, THTensor *output, THTensor *weight, THTensor *bias, THTensor *finput,
                                                         int kW, int kH, int dW, int dH, int padW, int padH,
                                                         long nInputPlane, long inputWidth, long inputHeight,
                                                         long nOutputPlane, long outputWidth, long outputHeight)
{
  long i;

  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(output->storage->data+output->storageOffset+output->stride[0]*i, THTensor_(get1d)(bias, i), outputHeight*outputWidth);

#ifdef BLASCONV
  THTensor_(convmm)(output, 1, 1, weight, input, kH, kW, dH, dW, padH, padW, 0);
#else
  THTensor *output2d;

  THNN_(unfolded_copy)(finput, input, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight);

  output2d = THTensor_(newWithStorage2d)(output->storage, output->storageOffset,
                                         nOutputPlane, -1,
                                         outputHeight*outputWidth, -1);

  THTensor_(addmm)(output2d, 1, output2d, 1, weight, finput);

  THTensor_(free)(output2d);
#endif
}

void THNN_(SpatialConvolutionMM_updateOutput)(THNNState *state, THTensor *input, THTensor *output, THTensor *weight, THTensor *bias, THTensor *finput, THTensor* fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)
{
  int dimf = 0;
  int dimw = 2;
  int dimh = 1;

  long nInputPlane;
  long inputWidth;
  long inputHeight;
  long nOutputPlane;
  long outputWidth;
  long outputHeight;

  THArgCheck( input->nDimension == 3 || input->nDimension == 4, 1, "3D or 4D (batch mode) tensor expected");

  if (input->nDimension == 4) {
    dimf++;
    dimw++;
    dimh++;
  }
  if(weight->nDimension == 4)
    THTensor_(resize2d)(weight, weight->size[0], weight->size[1] * weight->size[2] * weight->size[3]);
  nInputPlane = input->size[dimf];
  inputWidth   = input->size[dimw];
  inputHeight  = input->size[dimh];
  nOutputPlane = weight->size[0];
  outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
        nInputPlane,inputHeight,inputWidth,nOutputPlane,outputHeight,outputWidth);

  if (nInputPlane*kW*kH != weight->size[1])
    THError("Wrong number of input channels! Input has %d channels, expected %d",nInputPlane,weight->size[1]/(kW*kH));

  if(input->nDimension == 3)
  {
    THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

    THNN_(SpatialConvolutionMM_updateOutput_frame)(input, output, weight, bias, finput,
                                                 kW, kH, dW, dH, padW, padH,
                                                 nInputPlane, inputWidth, inputHeight,
                                                 nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

    THTensor_(resize3d)(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize4d)(output, T, nOutputPlane, outputHeight, outputWidth);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

      THNN_(SpatialConvolutionMM_updateOutput_frame)(input_t, output_t, weight, bias, finput_t,
                                                   kW, kH, dW, dH, padW, padH,
                                                   nInputPlane, inputWidth, inputHeight,
                                                   nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(finput_t);
    }
  }
}


static void THNN_(SpatialConvolutionMM_updateGradInput_frame)(THTensor *gradInput, THTensor *gradOutput, THTensor *weight, THTensor *fgradInput,
                                                            int kW, int kH, int dW, int dH, int padW, int padH)
{
#ifdef BLASCONV
  if(dW == 1 && dH == 1)
  {
    THTensor *tweight = THTensor_(newTranspose)(weight,0,1);
    THTensor_(convmm)(gradInput, 0, 1, tweight, gradOutput, kH, kW, 1, 1, padH, padW, 1);
	THTensor_(free)(tweight);
  } else
#endif
  {
    THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                                         gradOutput->size[0], -1,
                                                         gradOutput->size[1]*gradOutput->size[2], -1);
    THTensor_(addmm)(fgradInput, 0, fgradInput, 1, weight, gradOutput2d);
    THTensor_(free)(gradOutput2d);

    THTensor_(zero)(gradInput);

    THNN_(unfolded_acc)(fgradInput, gradInput, kW, kH, dW, dH, padW, padH, gradInput->size[0], gradInput->size[2], gradInput->size[1], gradOutput->size[2], gradOutput->size[1]);
  }
}

void THNN_(SpatialConvolutionMM_updateGradInput)(THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput, THTensor *weight, THTensor *bias, THTensor *finput, THTensor *fgradInput, int kW, int kH, int dW, int dH, int padW, int padH)
{
  long nOutputPlane = weight->size[0];

  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);
  THTensor_(transpose)(weight, weight, 0, 1);

  if(input->nDimension == 3)
  {
    THNN_(SpatialConvolutionMM_updateGradInput_frame)(gradInput, gradOutput, weight, fgradInput, kW, kH, dW, dH, padW, padH);
  }
  else
  {
    long T = input->size[0];
    long t;

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

      THNN_(SpatialConvolutionMM_updateGradInput_frame)(gradInput_t, gradOutput_t, weight, fgradInput_t, kW, kH, dW, dH, padW, padH);

      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
      THTensor_(free)(fgradInput_t);
    }
  }

  THTensor_(transpose)(weight, weight, 0, 1);
}

static void THNN_(SpatialConvolutionMM_accGradParameters_frame)(THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias, THTensor *finput,
                                                              THTensor *input, int kW, int kH, int dW, int dH, int padW, int padH, real scale)
{
  long i;
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                                       gradOutput->size[0], -1,
                                                       gradOutput->size[1]*gradOutput->size[2], -1);

#ifdef BLASCONV
  THTensor_(convmm)(gradWeight, 1, scale, gradOutput, input, kH, kW, dH, dW, padH, padW, 2);
#else
  THTensor_(transpose)(finput, finput, 0, 1);
  THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput2d, finput);
  THTensor_(transpose)(finput, finput, 0, 1);
#endif

  for(i = 0; i < gradBias->size[0]; i++)
  {
    long k;
    real sum = 0;
    real *data = gradOutput2d->storage->data + gradOutput2d->storageOffset + i*gradOutput2d->stride[0];
    for(k = 0; k < gradOutput2d->size[1]; k++)
      sum += data[k];
    (gradBias->storage->data + gradBias->storageOffset)[i] += scale*sum;
  }

  THTensor_(free)(gradOutput2d);
}

void THNN_(SpatialConvolutionMM_accGradParameters)(THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias, THTensor *finput, THTensor *fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, real scale)
{
  long nOutputPlane = gradWeight->size[0];
  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

  if(input->nDimension == 3)
  {
    THNN_(SpatialConvolutionMM_accGradParameters_frame)(gradOutput, gradWeight, gradBias, finput, input, kW, kH, dW, dH, padW, padH, scale);
  }
  else
  {
    long T = input->size[0];
    long t;

    for(t = 0; t < T; t++)
    {
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);
	  THTensor *input_t = THTensor_(newSelect)(input, 0, t);

      THNN_(SpatialConvolutionMM_accGradParameters_frame)(gradOutput_t, gradWeight, gradBias, finput_t, input_t, kW, kH, dW, dH, padW, padH, scale);

      THTensor_(free)(gradOutput_t);
      THTensor_(free)(finput_t);
	  THTensor_(free)(input_t);
    }
  }
}

#endif
