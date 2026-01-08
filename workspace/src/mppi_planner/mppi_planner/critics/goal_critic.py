#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import functools

class GoalCritic:
    def __init__(self, cfg,euclid_dim):
        self.enabled = cfg["enabled"]
        self.weight = cfg["cost_weight"]
        self.power = cfg["cost_power"]
        self.threshold = cfg["threshold_to_consider"]
        self.euclid_dim = euclid_dim
    @functools.partial(jax.jit, static_argnums=0)
    def __call__(self, state, goal):
        if not self.enabled:
            return jnp.zeros((state.shape[0],))

        dist = jnp.linalg.norm(state[:, :self.euclid_dim] - goal[:self.euclid_dim,0], axis=1)
        cost = jnp.where(
            dist > self.threshold,
            (dist ** self.power) * self.weight,
            0.0,
        )
        return cost
