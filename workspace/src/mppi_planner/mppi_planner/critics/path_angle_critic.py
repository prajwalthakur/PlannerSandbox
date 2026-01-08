#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import functools


class PathAngleCritic:
    def __init__(self, cfg,dim_euclid):
        self.enabled = cfg["enabled"]
        self.weight = cfg["cost_weight"]
        self.power = cfg["cost_power"]
        self.offset_from_furthest = cfg["offset_from_furthest"]
        self.threshold = cfg["threshold_to_consider"]
        self.max_angle_to_furthest = cfg["max_angle_to_furthest"]
        self.mode = cfg["mode"]
        self.dim_euclid = dim_euclid
    @functools.partial(jax.jit, static_argnums=0)
    def __call__(self, rbt_state, rbt_goal):
        euclid_dim = self.dim_euclid
        if not self.enabled:
            return jnp.zeros((rbt_state.shape[0],))
        dist = jnp.linalg.norm(rbt_state[:, :euclid_dim] - rbt_goal[:euclid_dim,0], axis=1)
        dist_mask = dist > self.threshold
        if(self.mode==0):
            anlge_allign_cost = self.forward_preference(rbt_state,rbt_goal)

            cost = jnp.where(dist_mask, anlge_allign_cost, 0.0)
        
        return cost
    def forward_preference(self,rbt_state,rbt_goal):

        dx =  rbt_goal[0] - rbt_state[:,0]
        dy = rbt_goal[1] - rbt_state[:,1]
        desired_yaw = jnp.arctan2(dy,dx)

        # rbt yaw
        rbt_yaw = rbt_state[:,-1]
        
        #  Angle error (wrapped) 
        yaw_error = desired_yaw - rbt_yaw
        yaw_error = jnp.arctan2(jnp.sin(yaw_error), jnp.cos(yaw_error))
        abs_yaw_error = jnp.abs(yaw_error)

        # Penalize large misalignment 
        angle_cost = jnp.where(
            abs_yaw_error > self.max_angle_to_furthest,
            self.weight * (abs_yaw_error ** self.power),
            0.0
            )
        return angle_cost