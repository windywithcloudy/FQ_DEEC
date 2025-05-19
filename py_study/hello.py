import torch
x = torch.arange(4.0)
x.requires_grad_(True)
y = x.sum()
y.backward()
print(x.grad)
x.grad.zero_()
print(torch.cuda.is_available())