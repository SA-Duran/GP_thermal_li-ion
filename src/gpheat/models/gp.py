from typing import Any, Dict
import torch
import gpytorch

class _GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPRegressor:
    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = None

    def fit(self, X, y, iters: int = 30):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.model = _GPModel(X, y, self.likelihood)
        self.model.train(); self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(iters):
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        self.model.eval(); self.likelihood.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(X))
        return preds.mean.numpy()
