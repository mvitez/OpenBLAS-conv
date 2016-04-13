require 'nn'

function fill(t)
   s = t:storage()
   for i=1,s:size() do
      s[i] = i
   end
end

dir = 'ref'
torch.setdefaulttensortype('torch.FloatTensor')
m = nn.SpatialConvolution(64, 48, 9, 5, 1, 1, 2, 4)
fill(m.weight)
fill(m.bias)
fill(m.gradBias)
fill(m.gradWeight)
i = torch.Tensor(64,128,96)
fill(i)
sys.tic()
out = m:forward(i)
print(sys.toc())
torch.save(dir .. '/out.t7', out)
gradOutput = torch.Tensor(out:size(1), out:size(2), out:size(3))
fill(gradOutput)
sys.tic()
gradInput = m:updateGradInput(i, gradOutput)
print(sys.toc())
torch.save(dir .. '/gradInput.t7', gradInput)
sys.tic()
gradWeight = m:accGradParameters(i, gradOutput, 1)
print(sys.toc())
torch.save(dir .. '/gradBias.t7', m.gradBias)
torch.save(dir .. '/gradWeight.t7', m.gradWeight)

