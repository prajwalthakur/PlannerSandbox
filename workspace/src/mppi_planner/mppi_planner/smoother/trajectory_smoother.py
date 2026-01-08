#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import functools
class TrajectorySmoother:
    def __init__(self, cfg, dim):
        self.window_size = cfg["window_size"]
        self.poly_order = cfg["poly_order"]
        self.dim = dim

        # Compute SG kernel ONCE
        k1d = self.savgol_kernel(self.window_size, self.poly_order )

        # Expand for depthwise conv: (KW, 1, C)
        self.kernel = jnp.tile(
            k1d[:, None, None],
            (1, 1, dim)
        )
    def savgol_kernel(self,window_size: int, poly_order: int):
        assert window_size % 2 == 1, "window_size must be odd"
        assert poly_order < window_size

        half = window_size // 2
        x = jnp.arange(-half, half + 1)

        # Vandermonde matrix
        A = jnp.vander(x, poly_order + 1, increasing=True)

        # Least-squares solution
        ATA_inv = jnp.linalg.pinv(A.T @ A)
        coeffs = ATA_inv @ A.T

        # Take the row corresponding to the 0th derivative at center
        kernel = coeffs[0]

        return kernel
    @functools.partial(jax.jit, static_argnums=0)
    def __call__(self, xx):
        # xx: (T, dim)
        x = xx[None, :, :]

        y = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.kernel,
            window_strides=(1,),
            padding="SAME",
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=self.dim
        )
        return y[0]
