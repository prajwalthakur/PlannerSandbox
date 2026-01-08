#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import functools

class GoalAngleCritic:
    def __init__(self, cfg,euclid_dim):
        self.enabled = cfg["enabled"]
        self.weight = cfg["cost_weight"]
        self.power = cfg["cost_power"]
        self.threshold = cfg["threshold_to_consider"]
        self.euclid_dim =  euclid_dim
    @functools.partial(jax.jit, static_argnums=0)
    def __call__(self, state, goal):
        if not self.enabled:
            return jnp.zeros((state.shape[0],))

        euclid_dist = jnp.linalg.norm(state[:, :self.euclid_dim] - goal[:self.euclid_dim,0], axis=1)
        rbt_yaw = state[:,-1]
        goal_yaw = goal[None,-1,0]
        diff_yaw = (rbt_yaw - goal_yaw)
        orient_error = jnp.arctan2(jnp.sin(diff_yaw),jnp.cos(diff_yaw))
        
        cost = jnp.where(
            euclid_dist < self.threshold,
            (jnp.abs(orient_error) ** self.power) * self.weight,
            0.0,
        )
        return cost
