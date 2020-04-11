import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# start torch
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_points = 1000
X = torch.randn(n_points, 2).to(device)
colors = X[:, 0]

# SVD
OI = torch.cat((torch.zeros(2, 2), torch.eye(2))).to(device)
for i in range(10):
    W = torch.randn(2, 2).to(device)
    Y = X @ W.t()
    U, S, V = torch.svd(W)
    new_OI = OI @ W

# linear transformation
model = nn.Sequential(nn.Linear(2, 2, bias=False))
model.to(device)
with torch.no_grad:
    Y = model(X)

# non-linear transformation
z = torch.linspace(-10, 10, 101)
s = torch.tanh(z)
model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Tanh())
model.to(device)
for s in range(5):
    W = s * torch.eye(2)
    model[0].weight.data.copy_(W)
    Y = model(X).data

# NN
n_hidden = 5
NL = nn.Tanh()
for i in range(5):
    model = nn.Sequential(
        nn.Linear(2, n_hidden),
        NL,
        nn.Linear(n_hidden, 2)
    )
    model.to(device)
    with torch.no_grad():
        Y = model(X)

# deeper NN
for i in range(5):
    model = nn.Sequential(
        nn.Linear(2, n_hidden),
        NL,
        nn.Linear(n_hidden, n_hidden),
        NL,
        nn.Linear(n_hidden, n_hidden),
        NL,
        nn.Linear(n_hidden, 2),
    )
    model.to(device)
    with torch.no_grad():
        Y = model(X).detach()