ref_out = torch.load('ref/out.t7')
ref_gradInput = torch.load('ref/gradInput.t7')
ref_gradWeight = torch.load('ref/gradWeight.t7')
ref_gradBias = torch.load('ref/gradBias.t7')

new_out = torch.load('new/out.t7')
new_gradInput = torch.load('new/gradInput.t7')
new_gradWeight = torch.load('new/gradWeight.t7')
new_gradBias = torch.load('new/gradBias.t7')

print(torch.max(new_out - ref_out))
print(torch.max(new_gradInput - ref_gradInput)/torch.max(ref_gradInput))
print(torch.max(new_gradWeight - ref_gradWeight))
print(torch.max(new_gradBias - ref_gradBias))