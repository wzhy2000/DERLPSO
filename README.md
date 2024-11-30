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

#### 1）Defining differential equations

```
def lotka_volterra(state, t, α, β, γ, δ):
    l, v = state
    dxdt = α * l - β * l * v
    dydt = δ * l * v - γ * v
    return [dxdt, dydt]
```


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

For your convenience, we provide an ODE（ODE_Demo.py） and a PDE（PDE_Demo.py） example respectively, which introduces how to use this model.

## How to cite

(1) Sun, W. K., Fan X. Y., Jia, L. J., Chu. T., Yau, S. T., Wu, R. L., & Wang Z. (2024). Estimating unknown parameters in differential equations with a reinforcement learning based PSO method. (https://doi.org/10.48550/arXiv.2411.08651)
