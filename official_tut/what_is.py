from __future__ import print_function
import torch


x = torch.Tensor(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

print(x.size())

if torch.cuda.is_available():
    print("avail")
