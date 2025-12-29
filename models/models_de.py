
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class MLPDiagGaussian(nn.Module):
    """
    Heteroscedastic diagonal Gaussian regressor:
      p(y|x) = N(mu(x), diag(sigma(x)^2))
    """
    def __init__(self, xdim: int, ydim: int, hidden: int = 256, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_f = xdim
        for _ in range(depth):
            layers += [nn.Linear(in_f, hidden), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_f = hidden
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden, ydim)
        self.logstd_head = nn.Linear(hidden, ydim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu_head(h)
        logstd = self.logstd_head(h).clamp(-10.0, 5.0)
        return mu, logstd


def nll_diag_gaussian(y: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    """
    Mean NLL over batch, summed over dims.
    """
    var = torch.exp(2.0 * logstd)
    return 0.5 * torch.sum((y - mu) ** 2 / var + 2.0 * logstd + math.log(2.0 * math.pi), dim=-1).mean()


@torch.no_grad()
def eval_single(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 8192) -> Dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    x = x.to(device)
    y = y.to(device)

    rmses, nlls = [], []
    for i0 in range(0, x.shape[0], batch_size):
        xb = x[i0:i0 + batch_size]
        yb = y[i0:i0 + batch_size]
        mu, logstd = model(xb)

        rmse = torch.sqrt(torch.mean((mu - yb) ** 2, dim=-1))  # (B,)
        rmses.append(rmse.detach().cpu())

        var = torch.exp(2.0 * logstd)
        nll = 0.5 * torch.sum((yb - mu) ** 2 / var + torch.log(var) + math.log(2.0 * math.pi), dim=-1)
        nlls.append(nll.detach().cpu())

    return {"rmse": torch.cat(rmses).mean().item(), "nll": torch.cat(nlls).mean().item()}


@torch.no_grad()
def eval_ensemble(models: List[nn.Module], x: torch.Tensor, y: torch.Tensor, batch_size: int = 8192) -> Dict[str, float]:
    """
    Ensemble predictive mean/variance for mixture of diag Gaussians:
      mu_ens = mean(mu_i)
      var_ens = mean(var_i) + Var(mu_i)
    """
    device = next(models[0].parameters()).device
    for m in models:
        m.eval()

    x = x.to(device)
    y = y.to(device)

    rmses, nlls = [], []
    for i0 in range(0, x.shape[0], batch_size):
        xb = x[i0:i0 + batch_size]
        yb = y[i0:i0 + batch_size]

        mus, vars_ = [], []
        for m in models:
            mu, logstd = m(xb)
            mus.append(mu)
            vars_.append(torch.exp(2.0 * logstd))

        mus = torch.stack(mus, dim=0)     # (M,B,D)
        vars_ = torch.stack(vars_, dim=0) # (M,B,D)

        mu_ens = mus.mean(dim=0)
        var_ens = vars_.mean(dim=0) + mus.var(dim=0, unbiased=False)

        rmse = torch.sqrt(torch.mean((mu_ens - yb) ** 2, dim=-1))
        rmses.append(rmse.detach().cpu())

        nll = 0.5 * torch.sum((yb - mu_ens) ** 2 / var_ens + torch.log(var_ens) + math.log(2.0 * math.pi), dim=-1)
        nlls.append(nll.detach().cpu())

    return {"rmse": torch.cat(rmses).mean().item(), "nll": torch.cat(nlls).mean().item()}
