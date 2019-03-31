#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn & Jiyuan Zhou

Environment for training local planner to move on a straight way.
"""

import csv

import numpy as np

from rllab.envs.base import Step
from rllab.spaces import Box

from aa_simulation.envs.base_env import VehicleEnv


class LocalPlannerEnvStraightRandDirection(VehicleEnv):
    """
    Simulation environment for an RC car following a circular
    arc trajectory using relative coordinates.
    """

    def __init__(self, initial_position_x,\
                        initial_position_y, initial_direction,\
                        initial_velocity_dx, initial_velocity_dy,\
                        target_position_x, target_position_y,\
                        target_velocity):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(LocalPlannerEnvStraightRandDirection, self).__init__(target_velocity)

        # Parameters of the line to follow
        self.init_x = initial_position_x
        self.init_y = initial_position_y
        self.init_dir = initial_direction
        self.init_dx = initial_velocity_dx
        self.init_dy = initial_velocity_dy

        self.target_x = target_position_x
        self.target_y = target_position_y

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(6,))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        state = np.zeros(6)
        state[0] = self.init_x
        state[1] = self.init_y
        state[2] = np.deg2rad(self.init_dir)
        state[3] = self.init_dx
        state[4] = self.init_dy
        return state


    def step(self, action):
        """
        Move one iteration forward in simulation.
        """
        # Get next state from dynamics equations
        self._action = action
        nextstate = self._model.state_transition(self._state, action,
                self._dt)

        # Check collision and assign reward to transition
        collision = self._check_collision(nextstate)
        if collision:
            reward = -100
            done = True
            distance = np.inf
            vel_diff = np.inf
        else:
            self._state = nextstate
            done = False

            # Trajectory following
            x, y, _, x_dot, y_dot, _ = nextstate

            # Velocity difference
            lambda1 = 1
            lambda2 = 2
            velocity = np.sqrt(np.square(x_dot) + np.square(y_dot))
            vel_diff = velocity - self.target_velocity

            # Position difference
            distance = self._cal_distance(x, y)
        
            if self._same_direction(x, y):
                x_rest = 0
            else:
                x_rest = self._dest_dist(x, y)

            reward = - lambda2 * np.abs(distance) - x_rest
            reward -= lambda1 * np.square(vel_diff)

        return Step(observation=nextstate, reward=reward,
                done=done, dist=distance, vel=vel_diff)


    def _same_direction(self, x, y):
        return (x - self.init_x) * (self.target_x - self.init_x)\
                + (y - self.init_y) * (self.target_y - self.init_y) > 0

    def _dest_dist(self, x, y):
        return (self.target_x - x) * (self.target_x - self.init_x)\
                    + (self.target_y - y) * (self.target_y - self.init_y)

    def _cal_distance(self, x, y):
        return np.abs((self.target_y - self.init_x) * x +\
                        (self.init_x - self.target_x) * y\
                        - self.init_x * self.target_y +\
                        self.target_x * self.init_y)\
                    / np.sqrt(np.square(self.target_y - self.init_y)\
                      + np.square(self.target_x - self.init_x))

    def reset(self):
        """
        Reset environment back to original state.
        """
        self._action = None
        self._state = self.get_initial_state
        observation = self._state

        # Reset renderer if available
        if self._renderer is not None:
            self._renderer.reset()

        return observation
