require 'nn'

function fill(t)
   s = t:storage()
   for i=1,s:size() do
      s[i] = i
   end
end

torch.setdefaulttensortype('torch.FloatTensor')
m = nn.SpatialConvolution(64, 48, 9, 9, 1, 1, 4, 4)
fill(m.weight)
fill(m.bias)
fill(m.gradBias)
fill(m.gradWeight)
i = torch.Tensor(64,128,128)
fill(i)
print('input',i)
out = m:forward(i)
print('output',out)

gradOutput = torch.Tensor(out:size(1), out:size(2), out:size(3))
fill(gradOutput)
gradInput = m:updateGradInput(i, gradOutput)
print('gradInput',gradInput)
gradWeight = m:accGradParameters(i, gradOutput, 1)
--print('gradBias',m.gradBias)
--print('gradWeight',m.gradWeight)

