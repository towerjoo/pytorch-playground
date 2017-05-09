import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)
inputs = [autograd.Variable(torch.randn((1, 3))) for _ in range(5)]
inputs = torch.cat(inputs).view(len(inputs), 1, -1)

hidden = (autograd.Variable(torch.randn((1, 1, 3))),
          autograd.Variable(torch.randn((1, 1, 3))))
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)
