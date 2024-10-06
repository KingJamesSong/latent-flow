import torch
import torch.nn as nn
from torch.func import functional_call
from torch.func import vmap
from torch.func import jacrev


def div(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def build_divfree_vector_field(module):
    """Returns an unbatched vector field, i.e. assumes input is a 1D tensor."""

    #F_fn, params = functional_call(module)

    #torch.func.functional_call(module, params, (inputs,))

    params = dict(module.named_parameters())

    #J_fn = jacrev(functional_call(module, params, (inputs,)), argnums=1)
    def F_fn(params,x):
        out = functional_call(module, params, (x,))
        return out

    def A_fn(params, x):
        Jfn = jacrev(F_fn,argnums=1)
        J = Jfn(params, x)
        A = J - J.T
        return A

    def A_flat_fn(params, x):
        A = A_fn(params, x)
        A_flat = A.reshape(-1)
        return A_flat

    def ddF(params, x):
        D = x.nelement()
        dA_flat = jacrev(A_flat_fn, argnums=1)(params, x)
        Jac_all = dA_flat.reshape(D, D, D)
        ddF = vmap(torch.trace)(Jac_all)
        return ddF

    return ddF, params, A_fn, F_fn
