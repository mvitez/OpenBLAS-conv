require 'openblas-conv'

torch.setdefaulttensortype('torch.FloatTensor')

function fill(t)
   s = t:storage()
   for i=1,s:size() do
      s[i] = i % 100 * 0.01
   end
end

function test(enable, partial)
   openblasconv(enable, partial)
   input = torch.Tensor(3,720,1280)
   n = nn.SpatialConvolutionMM(3,8,5,5,1,1,2,2)
   fill(input)
   fill(n.weight)
   fill(n.bias)
   fill(n.gradBias)
   fill(n.gradWeight)
   sys.tic()
   for i = 1,100 do
      output = n:forward(input)
   end
   print('forward',  enable, partial, output:sum(), sys.toc())
   gradOutput = torch.Tensor(output:size(1), output:size(2), output:size(3))
   fill(gradOutput)
   sys.tic()
   for i = 1,100 do
      gradInput = n:backward(input, gradOutput)
   end
   print('backward', enable, partial, gradInput:sum(), sys.toc())
   collectgarbage()
end

test(true)
test(true,true)
test(false)
