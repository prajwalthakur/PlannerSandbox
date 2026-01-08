#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import functools

class ObstaclesCritic:
    """
    Docstring for ObstaclesCritic
    Compute the minimum obstacle distance per rollout, 
    classify it into mutually exclusive regions, 
    and apply exactly one cost term per rollout.
    """
    def __init__(self, cfg,euclid_dim):
        self.enabled = cfg["enabled"]
        self.power = cfg["cost_power"]
        self.repulsion_weight = cfg["repulsion_weight"]
        self.critical_weight = cfg["critical_weight"]
        self.collision_cost = cfg["collision_cost"]
        self.collision_margin_distance = cfg["collision_margin_distance"]
        self.near_goal_distance = cfg["near_goal_distance"]
        self.robot_r = cfg["robot_radius"]
        self.obs_r = cfg["obstacles_radius"]
        self.buffer_r = cfg["buffer_radius"]
        self.consider_obs_density = cfg["consider_obs_density"]
        self.effective_radius = self.robot_r+self.obs_r
        self.euclid_dim = euclid_dim
    
    @functools.partial(jax.jit, static_argnums=0)
    def __call__(self, state, obs_state, goal):
        euclid_dim = self.euclid_dim
        if not self.enabled:
            return jnp.zeros((state.shape[0],))
        if not self.consider_obs_density:
            dist_to_goal = jnp.linalg.norm(state[:, :euclid_dim] - goal[:euclid_dim], axis=1) #(k,)
            relax_mask = dist_to_goal < self.near_goal_distance

            rbt_xy = state[:,0:euclid_dim] #k,2
            obs_xy = obs_state[:,0:euclid_dim] #(N,2)
            diff = rbt_xy[:,None,:] - obs_xy[None,:,:] #(k,N,2)
            rbt_dist_to_obs = jnp.linalg.norm(diff,axis=-1) #(k,N)   

            d_min = jnp.min(rbt_dist_to_obs,axis=-1) #(k,)     
            
            r_eff = self.effective_radius
            delta = self.collision_margin_distance

            # Region mask
            collision_mask = d_min < r_eff
            critical_mask = (d_min>=r_eff) & (d_min < r_eff + delta)
            soft_mask =  (d_min >= (r_eff + delta))
            
            #costs
            collision_cost = jnp.where(collision_mask,self.collision_cost,0.0)
            critical_cost = jnp.where(critical_mask,self.critical_weight*jnp.exp(-self.power*(d_min-r_eff)),0.0)
            soft_cost = jnp.where(soft_mask,self.repulsion_weight*jnp.exp(-self.power*(d_min-r_eff)),0.0)

            total_cost = collision_cost + critical_cost + soft_cost
            total_cost = jnp.where(relax_mask, 0.3 * total_cost, total_cost)

            return total_cost
        if self.consider_obs_density:
            # Being surrounded by many obstacles is more dangerous 
            # than being near just one.
            # -----------------------
            # Goal relaxation
            # -----------------------
            dist_to_goal = jnp.linalg.norm(
                state[:, :euclid_dim] - goal[:euclid_dim,0], axis=1
            )  # (K,)
            relax_mask = dist_to_goal < self.near_goal_distance

            # -----------------------
            # Distances to obstacles
            # -----------------------
            rbt_xy = state[:, :euclid_dim]          # (K, 2)
            obs_xy = obs_state[:, :euclid_dim]      # (N, 2)

            diff = rbt_xy[:, None, :] - obs_xy[None, :, :]   # (K, N, 2)
            dists = jnp.linalg.norm(diff, axis=-1)           # (K, N)

            r_eff = self.effective_radius
            delta = self.collision_margin_distance

            # -----------------------
            # Region masks (K, N)
            # -----------------------
            collision_mask = dists < r_eff
            critical_mask  = (dists >= r_eff) & (dists < r_eff + delta)
            soft_mask      = dists >= (r_eff + delta)

            # -----------------------
            # Per-obstacle costs
            # -----------------------
            collision_cost = collision_mask * self.collision_cost

            critical_cost = critical_mask * (
                self.critical_weight *
                jnp.exp(-self.power * (dists - r_eff))
            )

            soft_cost = soft_mask * (
                self.repulsion_weight *
                jnp.exp(-self.power * (dists - r_eff - delta))
            )

            # -----------------------
            # Sum over obstacles
            # -----------------------
            total_cost = jnp.sum(
                collision_cost + critical_cost + soft_cost,
                axis=1
            )  # (K,)

            # -----------------------
            # Near-goal relaxation
            # -----------------------
            #total_cost = jnp.where(relax_mask, 0.3 * total_cost, total_cost)

            return total_cost


