from typing import Dict

import torch
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform


def build_conditional_maf(ydim: int, xdim: int, n_layers: int = 8, hidden: int = 256) -> Flow:
    transforms = []
    for _ in range(n_layers):
        transforms.append(RandomPermutation(features=ydim))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=ydim,
                hidden_features=hidden,
                context_features=xdim,
                num_blocks=2,
                use_residual_blocks=True,
                random_mask=False,
                activation=torch.nn.functional.gelu,
                dropout_probability=0.0,
                batch_norm_within_blocks=False,
            )
        )
    transform = CompositeTransform(transforms)
    base = StandardNormal(shape=[ydim])
    return Flow(transform, base)


@torch.no_grad()
def eval_flow_nll(flow: Flow, x: torch.Tensor, y: torch.Tensor, batch_size: int = 8192) -> Dict[str, float]:
    device = next(flow.parameters()).device
    flow.eval()
    x = x.to(device)
    y = y.to(device)

    nlls = []
    for i0 in range(0, x.shape[0], batch_size):
        xb = x[i0:i0 + batch_size]
        yb = y[i0:i0 + batch_size]
        logp = flow.log_prob(inputs=yb, context=xb)  # (B,)
        nlls.append((-logp).detach().cpu())

    return {"nll": torch.cat(nlls).mean().item()}


@torch.no_grad()
def eval_flow_rmse_sampling(flow: Flow, x: torch.Tensor, y: torch.Tensor, n_samples: int = 32, batch_size: int = 4096) -> Dict[str, float]:
    """
    RMSE of predictive mean estimated by sampling:
      mu(x) ≈ mean_{s=1..S} y_s  with y_s ~ p(y|x)
    """
    device = next(flow.parameters()).device
    flow.eval()
    x = x.to(device)
    y = y.to(device)

    rmses = []
    for i0 in range(0, x.shape[0], batch_size):
        xb = x[i0:i0 + batch_size]
        yb = y[i0:i0 + batch_size]
        samples = flow.sample(num_samples=n_samples, context=xb)  # (S,B,D)
        mu = samples.mean(dim=0)                                 # (B,D)
        rmse = torch.sqrt(torch.mean((mu - yb) ** 2, dim=-1))     # (B,)
        rmses.append(rmse.detach().cpu())

    return {"rmse": torch.cat(rmses).mean().item()}
