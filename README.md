# DERLPSO

DERLPSO can solve the unknown parameters of differential equations. At the same time, DERLPSO can avoid the shortcomings of traditional numerical methods that are highly sensitive to initial values ​​and prone to falling into local optimality. The DERLPSO method performs well in solving unknown parameters of differential equations and has the advantages of high accuracy, strong versatility, and independence from initial values.

## Required software

- Python 3.8.0
- scipy 1.10.1 
- fipy 3.4.5

## Usage

### For ODE

Just include ODE.py in the code - in addition to the standard library, scipy is also required.

To use the model to estimate parameters, you need to:

`ODEModel`: Ordinary differential equation to be solved

`paramNum`: The number of unknown parameters in an ordinary differential equation

`data`: Curve data corresponding to ordinary differential equations

`time`: The time series corresponding to ordinary differential equations

(Refer to the `lotka-volterra` structure defined in ODE_Demo.py)

### For PDE

Just include PDE.py in the code - in addition to the standard library, scipy and fipy are also required.

To use the model to estimate parameters, you need to:

`PDEModel`: Partial differential equation to be solved, it is necessary to provide a structure for calculating state variable data based on parameters

`paramNum`: The number of unknown parameters in an ordinary differential equation

`data`: Curve data corresponding to ordinary differential equations

(Refer to the `heat` structure defined in PDE_Demo.py)

## Other settings

The following can be set in the Model:

`particleNum`: The number of particles in the particle swarm

`maxIter`: Maximum number of iterations for the algorithm

`layersList`: Optional set of layers

`upper`: Upper limit value for uniform distribution initialization and upper limit value for logarithmic initialization positive sampling

`lower`：The lower limit of logarithmically initialized positive sampling (for uniform distribution initialization, the lower limit is the negative value of the upper limit)

`threshold`: When half of the maximum iteration times are reached, if the error has not reached this threshold, the particle will be reinitialized

## Examples

#### 1) Import required packages

```
import numpy as np
from scipy.integrate import odeint
from ODE import Model
```

#### 2) Define ODEs structure

```
def fitzhugh_nagumo(state, t, a, b, c ,d):
    v, w = state
    dvdt = d * (v - v ** 3 / 3 + w - c)
    dwdt = (-1 / d) * (v - a + b * w)
    return [dvdt, dwdt]
```

#### 3) Simulate ODE time series data

```
time =  np.arange(0, 4, step=(4 - 0) / 10)
data = odeint(fitzhugh_nagumo, [0, 0], time, args=(a, b, c, d))
```

#### 4) Use the model to estimate parameters

```
model = Model(ODEModel=fitzhugh_nagumo, paramNum=4, data=data, time=time)
model.initParticles()
model.iterator()
```

#### 5) Get the estimated results

```
print(f"Param: {np.array([a, b, c, d])}")
print("Best:", ["{:.16f}".format(x) for x in model.getGBest()])
print(f"Fit: {model.getFit()}")

Param: [1.20074502 2.35474693 0.38332407 2.02531496]
Best: ['1.2007450195631340', '2.3547469310196574', '0.3833240710142340', '2.0253149605158263']
Fit: 7.414830285890858e-32
```

For your convenience, we provide an ODE（ODE_Demo.py） and a PDE（PDE_Demo.py） example respectively, which introduces how to use this model.

## How to cite

(1) Sun, W. K., Fan X. Y., Jia, L. J., Chu. T., Yau, S. T., Wu, R. L., & Wang Z. (2024). Estimating unknown parameters in differential equations with a reinforcement learning based PSO method. (https://doi.org/10.48550/arXiv.2411.08651)
