#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import functools

class DifferentialDrive:
    def __init__(self, cfg):
        self.euclid_dim =  cfg["dim_euclid"]
        self.control_limit = jnp.asarray(cfg["control_limit"])
        min_lin_acc_limit = float(cfg["min_acceleration"]["linear"])
        min_ang_acc_limit = float(cfg["min_acceleration"]["angular"])
        max_lin_acc_limit = float(cfg["max_acceleration"]["linear"])
        max_ang_acc_limit = float(cfg["max_acceleration"]["angular"])
        self.acceleration_limit  = jnp.asarray([[min_lin_acc_limit,max_lin_acc_limit],
                                                [min_ang_acc_limit,max_ang_acc_limit]
                                                ])
        self.dt = float(cfg["dt"])
        self.min_delta_vx =  self.dt*min_lin_acc_limit
        self.max_delta_vx =  self.dt*max_lin_acc_limit
        self.min_delta_ang = self.dt*min_ang_acc_limit
        self.max_delta_ang = self.dt*max_ang_acc_limit
        self.delta = jnp.asarray([[self.min_delta_vx,self.max_delta_vx],
                                  [self.min_delta_ang,self.max_delta_ang]
                                  ])

    @functools.partial(jax.jit, static_argnums=0)
    def control_clip(self,v: jnp.ndarray,min_val,max_val):
        return jnp.clip(v, min_val, max_val)
    @functools.partial(jax.jit,static_argnums=0)
    def acc_limit_filter(self,prev_u,cmd_u):
        """
        Docstring for acc_limit_filter
        
        :param self: Description
        :param cmd_ut: num_rolloutsxdim_ctrl
        :param prev_ut: num_rolloutsxdim_ctrl
        """
        prev_vx = prev_u[:,0]
        prev_wz = prev_u[:,1]
        cmd_vx = cmd_u[:,0]
        cmd_wz = cmd_u[:,1]
        # Linear velocity (sign-dependent)
        lower_vx = jnp.where(prev_vx>0.0,
                             prev_vx + self.min_delta_vx,
                             prev_vx - self.max_delta_vx)
        upper_vx = jnp.where(prev_vx > 0.0,
                             prev_vx + self.max_delta_vx,
                             prev_vx - self.min_delta_vx)
        new_vx = jnp.clip(cmd_vx,lower_vx,upper_vx)

        # Angular velocity 

        lower_wz = prev_wz - self.max_delta_ang
        upper_wz = prev_wz + self.max_delta_ang
        new_wz = jnp.clip(cmd_wz,lower_wz,upper_wz)

        new_cmd = jnp.column_stack([new_vx,new_wz])
        return new_cmd,new_cmd

    @functools.partial(jax.jit,static_argnums=0)
    def batched_acc_limit_filter(self, init_st,init_ut,cmd_ut,dt):
        """
        Docstring for acc_limit_filter
        
        :param self: Description
        :param init_st: num_rolloutsxdim_ctrl
        :param init_ut: num_rolloutsxdim_ctrl
        :param cmd_ut: num_rolloutsxTxdim_ctrl
        :param dt: float scalar
        : return filtered_cmd_u_seqs : num_rolloutsxTxdim_ctrl
        """
        _,filtered_cmd_u_seqs = jax.lax.scan(self.acc_limit_filter,init_ut,cmd_ut.transpose(1,0,2)) # (T, num_rollouts, dim_ctrl)
        return filtered_cmd_u_seqs.transpose(1,0,2) #(num_rollouts, T, dim_ctrl)
    
    @functools.partial(jax.jit, static_argnums=0)
    def batched_dynamics_step(self,st, ut, dt):
        v = ut[:, 0]
        w = ut[:, 1]
        theta = st[:, 2]

        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)

        Xdot = jnp.stack((dx, dy, w), axis=1)
        return st + Xdot * dt
    
    @functools.partial(jax.jit, static_argnums=0)
    def dynamics_step(self,st, ut, dt):
        v = ut[0]
        w = ut[1]
        theta = st[2]

        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)
        Xdot = jnp.array([dx, dy, w])
        return st + Xdot * dt  