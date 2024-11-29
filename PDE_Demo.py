
import numpy as np
from fipy import Grid1D, CellVariable, DiffusionTerm, TransientTerm
from scipy.stats import truncnorm

from PDE import Model

def heat(X):
    alpha = X[0]
    nx = 20
    Lx = 1.0
    mesh = Grid1D(nx=nx, Lx=Lx)
    T = CellVariable(name="temperature", mesh=mesh, value=0.0, hasOld=True)
    T.setValue([[0.1, 0.2, 0.5, 1, 1, 1, 1, 1, 0.5, 0.2, 0.2, 0.5, 1, 1, 1, 1, 1, 0.5, 0.2, 0.1]])
    T.constrain(0.0, mesh.facesLeft)
    T.constrain(0.0, mesh.facesRight)
    eq = TransientTerm() == DiffusionTerm(coeff=alpha)
    Lt = 1.0
    tx = 20
    dt = Lt / tx
    u = np.zeros((tx + 1, nx))
    u[0,] = T.value
    for step in range(tx):
        T.updateOld()
        eq.solve(var=T, dt=dt)
        u[step + 1,] = T.value
    return u


mu = 0.5
sigma = 0.5
lower, upper = 0.0001, 1
a, b = (lower - mu) / sigma, (upper - mu) / sigma
truncated_normal_dist = truncnorm(a, b, loc=mu, scale=sigma)
a = truncated_normal_dist.rvs()

data = heat([a])

model = Model(heat, 1, data)
model.initParticles()
model.iterator()

print(f"Param: {[a]}")
print("Best:", ["{:.16f}".format(x) for x in model.getGBest()])
print(f"Fit: {model.getFit()}")




