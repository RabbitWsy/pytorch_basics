import torch
import time

print(torch.cuda.is_available())

how_long = 10000

a = torch.arange(how_long).cuda()

# speech
start_time = time.time()
a = a + 1
print(a)
print(time.time() - start_time)

# construction
b = torch.tensor([[1, 2], [3, 4]])
print(b.dim(), b.shape)

b = torch.zeros(3, 2)
print(b)

b = torch.ones(3, 2)
print(b)

b = torch.randn(3, 2)
print(b)

# dtype
b = torch.tensor([1, 2, 3], dtype=torch.float16)
print(b)

# slice
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a1 = a[0:2, 1:3]
print(a1)

idx = [1, 1, 0, 2, 2]
a2 = a[:, idx]
print(a2)

a[:, 1] = 0
print(a)

# reshape
a = torch.arange(9)
print(a)
a1 = a.view(3, 3)
print(a1)
a2 = a.view(-1, 3)
print(a2)

a3 = a1.t()
print(a3)

b = torch.rand(3, 2, 4)
print(b)
b1 = b.permute(1, 2, 0)
print(b1)

# operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b
print(c)
c = a**b
print(c)
print(a.sin())
print(a.sqrt())

# reduction
c = torch.tensor([[1, 2, 3], [4, 5, 6]])
c1 = c.sum()
print(c1)
c2 = c.sum(dim=0)
print(c2)
c3 = c.sum(dim=1, keepdim=True)
print(c3)

print(c.min(dim=1))

# matrix multiplication
c = torch.tensor([[1, 2, 3], [4, 5, 6]])
d = torch.tensor([[7, 8, 9], [10, 11, 12]])
e = c @ d.t()
print(e)

# broadcasting
c = torch.tensor([[1, 2, 3], [4, 5, 6]])
d = torch.tensor([1, 2, 3])
e = c + d
print(e)

# device
a = torch.tensor([1, 2, 3], dtype=torch.float32, device="cuda")
print(a)
a = a.cuda()
print(a)
a = a.cpu()
print(a)
a = a.to("cuda:1")
print(a)

# gradient
a = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
b = a**2
c = b.sum()
c.backward()
print(a.grad)
