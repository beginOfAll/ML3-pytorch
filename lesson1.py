import torch
import numpy as np
from torch.autograd import Variable

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
a = torch_data.numpy()
print(torch_data)
print(a)

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)
print(tensor)