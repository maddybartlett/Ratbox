import numpy as np
import os
from .steering import SteeringModel

from gymnasium import spaces

from pygame.math import Vector2
import pygame


class SkidSteer(SteeringModel):
    '''
    Skid Steer model (a.k.a. tank driving), state space is (x, y, heading)
    Commands are [left speed, right speed]
    Assuming a z-up coordinate system
    '''
    def __init__(self): 
        super().__init__(state_dim=3, command_dim=2, max_speed=100)

        self._action_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32), 
                                        high=np.array([1, 1], dtype=np.float32)
                                        )

        self._observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, 0], dtype=np.float32), 
                                             high=np.array([np.inf, np.inf, 2 * np.pi], dtype=np.float32)
                                             )
    def action_space(self):
        return self._action_space

    def observation_space(self):
        return self._observation_space
    
    def step(self, agent, u, dt=1, x=None):
        
        ## load agent sprite and get width
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '..', 'utils'))
        path = os.path.join(__location__, f'assets\\rat.png')
        image = pygame.image.load(path)
        self._width = image.get_height()

        ## get agent state
        x = np.array([agent.position[0], agent.position[1], np.deg2rad(agent.direction+90)])

        ## check action and state have correct number of elements
        assert len(u) == 2, f'Expected 2 action commands, got {len(u)}'
        assert len(x) == 3, f'Expected 3 items in state, got {len(x)}'

        ## calculate velocity as half the sum of the speeds of the left and right tracks
        vel = ((u[0] + u[1]) / 2) * self.max_speed()
        ## calculate rotation as the diff between track speeds divided by the agent width
        dtheta = (u[1] - u[0]) / self._width
        
        ## Calculate displacement
        dx = np.zeros(x.shape)
        dx[:2] = vel * np.array([np.sin(x[2]), np.cos(x[2])])
        dx[2] = dtheta
        
        ## Calculate new location (x, y)
        new_state = x + dx * dt
        new_pos = list(new_state[:2])
        
        ## Get agent's new direction/heading
        new_dir = new_state[2]%(2*np.pi)
        
        return Vector2(new_pos), np.rad2deg(new_dir)-90
    