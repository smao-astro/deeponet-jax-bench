"""
JAX implementation of DeepONet
"""

import jax.numpy as jnp
from flax import linen as nn


class FNN(nn.Module):
    features: tuple

    def setup(self):
        # noinspection PyAttributeOutsideInit
        self.layers = [nn.Dense(name=f'dense_{i}', features=feat,
                                kernel_init=nn.initializers.glorot_normal(),
                                bias_init=nn.initializers.zeros) for
                       i, feat in enumerate(self.features[1:])]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.tanh(layer(x))
        x = self.layers[-1](x)
        return x


class DeepONet(nn.Module):
    branch_features: tuple
    trunk_features: tuple
    cartesian_prod: bool = True

    def setup(self):
        # noinspection PyAttributeOutsideInit
        self.branch, self.trunk, self.bias = (
            FNN(self.branch_features),
            FNN(self.trunk_features),
            self.param('bias', nn.initializers.zeros, ())
        )

    def __call__(self, branch_in, trunk_in):
        """

        (Lp,), (Lx,) -> () # for double vmap
        (M, Lp), (Lx,) -> (M,) # for single vmap
        (Lp,), (N, Lx,) -> (N,) # for single vmap
        cartesian_prod==False: (MN, Lp), (MN, Lx) -> (MN,)
        cartesian_prod==True: (M, Lp), (N, Lx) -> (M, N)

        Returns:

        """
        # forward of branch and trunk
        branch_out = self.branch(branch_in)
        trunk_out = nn.tanh(self.trunk(trunk_in))  # only trunk output is activated before einsum
        if self.cartesian_prod and branch_in.ndim == 2 and trunk_in.ndim == 2:
            # (M, Lp), (N, Lx) -> (M, N)
            branch_out = jnp.expand_dims(branch_out, axis=1)
        out = jnp.sum(branch_out * trunk_out, axis=-1)
        out += self.bias
        # if out_channels is 1, squeeze this dimension
        return out
