import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
print (x)

y = x + 2

print y
print y.creator

z = y*y*3
out = z.mean()
print (z, out)

out.backward()
print (x.grad)


x = torch.randn(3)
x = Variable(x, requires_grad=True)
y = x*2
while y.data.norm() < 1000:
    y = y * 2

print(y)
