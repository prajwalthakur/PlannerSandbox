#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import functools


class PreferForwardCritic:
    def __init__(self, cfg,dim_euclid,ctrlTs):
        self.enabled = cfg["enabled"]
        self.weight = cfg["cost_weight"]
        self.power = cfg["cost_power"]
        self.threshold = cfg["threshold_to_consider"]
        self.dim_euclid = dim_euclid
        self.ctrlTs = ctrlTs
    @functools.partial(jax.jit, static_argnums=0)
    def __call__(self, rbt_state, rbt_ctrl, rbt_goal):
        euclid_dim =self.dim_euclid
        dt=self.ctrlTs
        if not self.enabled:
            return jnp.zeros((rbt_ctrl.shape[0],))
        cmd_vx = rbt_ctrl[:,0]

        dist = jnp.linalg.norm(rbt_state[:, :euclid_dim] - rbt_goal[:euclid_dim,0], axis=1)
        dist_mask = dist > self.threshold

        backward_cost = jnp.where( cmd_vx < 0, ((-cmd_vx*dt) ** self.power) * self.weight, 0.0)
        cost = jnp.where(dist_mask, backward_cost, 0.0)
        return cost
