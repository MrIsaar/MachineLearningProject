import math
import numpy as np
import torch

data = [[1,2],[3,4]]
nparr = np.array(data)
x_np = torch.from_numpy(nparr)
ones = torch.ones((2,3))
print(f"Tensor: \n {ones}\n")
tensor = torch.rand((2,3))
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")
print(tensor.device)