require 'nn'

-- Check what AccGradParameters is doing
require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')
m = nn.SpatialConvolution(1, 1, 2, 2, 1, 1, 1, 1)
m.weight[1][1][1][1] = 1
m.weight[1][1][1][2] = 2
m.weight[1][1][2][1] = 4
m.weight[1][1][2][2] = 8
m.bias[1] = 0
m.gradBias:zero()
m.gradWeight:zero()
i = torch.Tensor(1,4,4)
i:zero()
i[1][1][1] = 1
i[1][1][2] = 10
i[1][2][1] = 100
i[1][2][2] = 0
print('input',i)
out = m:forward(i)
print('output',out)

gradOutput = torch.Tensor(1,5,5):zero()
gradOutput[1][1][1] = 10
gradOutput[1][1][2] = 2
print('gradOutput', gradOutput)
gradInput = m:updateGradInput(i, gradOutput)
print('gradInput',gradInput)
gradWeight = m:accGradParameters(i, gradOutput, 1)
print('gradBias',m.gradBias)
print('gradWeight',m.gradWeight)

