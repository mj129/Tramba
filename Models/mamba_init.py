import torch
import math
import torch.nn as nn
from einops import repeat


def Dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
    dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = dt_rank ** -0.5 * dt_scale
    if dt_init == "constant":
        nn.init.constant_(dt_proj.weight, dt_init_std)
    elif dt_init == "random":
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
    else:
        raise NotImplementedError

    # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
    dt = torch.exp(
        torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inv_dt)
    # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
    # dt_proj.bias._no_reinit = True

    return dt_proj


def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
    # S4D real initialization
    A = repeat(
        torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
        "n -> d n",
        d=d_inner,
    ).contiguous()
    A_log = torch.log(A)  # Keep A_log in fp32
    if copies > 0:
        A_log = repeat(A_log, "d n -> r d n", r=copies)
        if merge:
            A_log = A_log.flatten(0, 1)
    A_log = nn.Parameter(A_log)
    A_log._no_weight_decay = True
    return A_log


def D_init(d_inner, copies=-1, device=None, merge=True):
    # D "skip" parameter
    D = torch.ones(d_inner, device=device)
    if copies > 0:
        D = repeat(D, "n1 -> r n1", r=copies)
        if merge:
            D = D.flatten(0, 1)
    D = nn.Parameter(D)  # Keep in fp32
    D._no_weight_decay = True
    return D
