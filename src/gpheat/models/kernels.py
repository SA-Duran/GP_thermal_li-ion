import torch
import gpytorch

def rbf_kernel():
    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
