from __future__ import annotations
import numpy as np
import GPy

# active_dims convention:
#   0 -> Vdot
#   1..4 -> other overpotential-related features

def kernel_multiplicative(fixed, input_dims: int) -> GPy.kern.Kern:
    kV = GPy.kern.RBF(input_dim=1, lengthscale=fixed["kV_lengthscale"],
                      variance=fixed["kV_variance"], ARD=False, active_dims=[0])
    k1 = GPy.kern.RBF(input_dim=input_dims-1, lengthscale=np.array(fixed["k1_lengthscales"]),
                      variance=fixed["k1_variance"], ARD=True, active_dims=[1,2,3,4])
    return kV * k1

def kernel_additive(fixed, input_dims: int) -> GPy.kern.Kern:
    kV = GPy.kern.RBF(input_dim=1, lengthscale=fixed["kV_lengthscale"],
                      variance=fixed["kV_variance"], ARD=False, active_dims=[0])
    k1 = GPy.kern.RBF(input_dim=input_dims-1, lengthscale=np.array(fixed["k1_lengthscales"]),
                      variance=fixed["k1_variance"], ARD=True, active_dims=[1,2,3,4])
    return kV + k1

def kernel_mixed_linear_times_exp(fixed, input_dims: int) -> GPy.kern.Kern:
    # Linear on kV (active_dims=[0]) + product of RBF(kV)*RBF(k1) (exp-quad)
    kV_lin = GPy.kern.Linear(input_dim=1, variances=1.0, active_dims=[0])
    kV_exp = GPy.kern.RBF(input_dim=1, lengthscale=fixed["kV_lengthscale"],
                          variance=fixed["kV_variance"], ARD=False, active_dims=[0])
    k1_exp = GPy.kern.RBF(input_dim=input_dims-1, lengthscale=np.array(fixed["k1_lengthscales"]),
                          variance=fixed["k1_variance"], ARD=True, active_dims=[1,2,3,4])
    return kV_lin + (kV_exp * k1_exp)

def build_kernel(kind: str, fixed: dict, input_dims: int) -> GPy.kern.Kern:
    if kind == "mult":
        return kernel_multiplicative(fixed, input_dims)
    if kind == "add":
        return kernel_additive(fixed, input_dims)
    if kind == "mixed":
        return kernel_mixed_linear_times_exp(fixed, input_dims)
    raise ValueError(f"Unknown kernel kind: {kind}")
