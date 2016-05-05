require 'nn'
local ffi = require 'ffi'

local virtmm = ffi.load('libopenblas-conv.so')

ffi.cdef [[
void THNN_DoubleSpatialConvolutionMM_updateOutput(
          THNNState *state,
          THDoubleTensor *input,
          THDoubleTensor *output,
          THDoubleTensor *weight,
          THDoubleTensor *bias,
          THDoubleTensor *finput,
          THDoubleTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
void THNN_DoubleSpatialConvolutionMM_updateGradInput(
          THNNState *state,
          THDoubleTensor *input,
          THDoubleTensor *gradOutput,
          THDoubleTensor *gradInput,
          THDoubleTensor *weight,
          THDoubleTensor *bias,
          THDoubleTensor *finput,
          THDoubleTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
void THNN_DoubleSpatialConvolutionMM_accGradParameters(
          THNNState *state,
          THDoubleTensor *input,
          THDoubleTensor *gradOutput,
          THDoubleTensor *gradWeight,
          THDoubleTensor *gradBias,
          THDoubleTensor *finput,
          THDoubleTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          double scale);
void THNN_FloatSpatialConvolutionMM_updateOutput(
          THNNState *state,
          THFloatTensor *input,
          THFloatTensor *output,
          THFloatTensor *weight,
          THFloatTensor *bias,
          THFloatTensor *finput,
          THFloatTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);

void THNN_FloatSpatialConvolutionMM_updateGradInput(
          THNNState *state,
          THFloatTensor *input,
          THFloatTensor *gradOutput,
          THFloatTensor *gradInput,
          THFloatTensor *weight,
          THFloatTensor *bias,
          THFloatTensor *finput,
          THFloatTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
void THNN_FloatSpatialConvolutionMM_accGradParameters(
          THNNState *state,
          THFloatTensor *input,
          THFloatTensor *gradOutput,
          THFloatTensor *gradWeight,
          THFloatTensor *gradBias,
          THFloatTensor *finput,
          THFloatTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          float scale);
]]

local saved = {}
saved.f1 = torch.getmetatable('torch.FloatTensor').THNN.SpatialConvolutionMM_updateOutput
saved.f2 = torch.getmetatable('torch.FloatTensor').THNN.SpatialConvolutionMM_updateGradInput
saved.f3 = torch.getmetatable('torch.FloatTensor').THNN.SpatialConvolutionMM_updateGradParameters
saved.d1 = torch.getmetatable('torch.DoubleTensor').THNN.SpatialConvolutionMM_updateOutput
saved.d2 = torch.getmetatable('torch.DoubleTensor').THNN.SpatialConvolutionMM_updateGradInput
saved.d3 = torch.getmetatable('torch.DoubleTensor').THNN.SpatialConvolutionMM_updateGradParameters

function openblasconv(enable, partial)
   if enable then
      torch.getmetatable('torch.FloatTensor').THNN.SpatialConvolutionMM_updateOutput =
         function(...) virtmm.THNN_FloatSpatialConvolutionMM_updateOutput(nil, ...) end
      if partial then
         torch.getmetatable('torch.FloatTensor').THNN.SpatialConvolutionMM_updateGradInput = saved.f2
      else
         torch.getmetatable('torch.FloatTensor').THNN.SpatialConvolutionMM_updateGradInput =
            function(...) virtmm.THNN_FloatSpatialConvolutionMM_updateGradInput(nil, ...) end
      end
      torch.getmetatable('torch.FloatTensor').THNN.SpatialConvolutionMM_updateGradParameters =
         function(...) virtmm.THNN_FloatSpatialConvolutionMM_updateGradParameters(nil, ...) end

      torch.getmetatable('torch.DoubleTensor').THNN.SpatialConvolutionMM_updateOutput =
         function(...) virtmm.THNN_DoubleSpatialConvolutionMM_updateOutput(nil, ...) end
      if partial then
         torch.getmetatable('torch.DoubleTensor').THNN.SpatialConvolutionMM_updateGradInput = saved.d2
      else
         torch.getmetatable('torch.DoubleTensor').THNN.SpatialConvolutionMM_updateGradInput =
            function(...) virtmm.THNN_DoubleSpatialConvolutionMM_updateGradInput(nil, ...) end
      end
      torch.getmetatable('torch.DoubleTensor').THNN.SpatialConvolutionMM_updateGradParameters =
         function(...) virtmm.THNN_DoubleSpatialConvolutionMM_updateGradParamaters(nil, ...) end
   else
      torch.getmetatable('torch.FloatTensor').THNN.SpatialConvolutionMM_updateOutput = saved.f1
      torch.getmetatable('torch.FloatTensor').THNN.SpatialConvolutionMM_updateGradInput = saved.f2
      torch.getmetatable('torch.FloatTensor').THNN.SpatialConvolutionMM_updateGradParameters = saved.f3
      torch.getmetatable('torch.DoubleTensor').THNN.SpatialConvolutionMM_updateOutput = saved.d1
      torch.getmetatable('torch.DoubleTensor').THNN.SpatialConvolutionMM_updateGradInput = saved.d2
      torch.getmetatable('torch.DoubleTensor').THNN.SpatialConvolutionMM_updateGradParameters = saved.d3
   end
end

openblasconv(true)
