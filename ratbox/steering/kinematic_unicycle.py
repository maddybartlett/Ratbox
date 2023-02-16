import numpy as np
from pygame.math import Vector2
from .steering import SteeringModel

from gymnasium import spaces
    

class KinematicUnicycle(SteeringModel):
    '''
    Kinematic unicycle, state space is (x, y, heading)
    Commands are [speed, direction]
    Assuming a z-up coordinate system
    '''
    def __init__(self):
        self._state_dim = 3
        self._command_dim = 2
        
        super().__init__(state_dim=self._state_dim, command_dim=self._command_dim, max_speed=100)

        self._action_space = spaces.Box(low=np.array([-1, -np.pi/2], dtype=np.float32), 
                                        high=np.array([1, np.pi/2], dtype=np.float32)
                                       )

        self._observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, 0], dtype=np.float32), 
                                             high=np.array([np.inf, np.inf, 2 * np.pi], dtype=np.float32)
                                             )
    def action_space(self):
        return self._action_space

    def observation_space(self):
        return self._observation_space
    
    def step(self, agent, u, dt=1, x=None):

        ## agent's state retrieved from agent class unless defined
        if x == None:
            x = np.array([agent.position[0], agent.position[1], 
                          np.deg2rad(agent.direction+90)])
        else:
            x = np.array([x[0], x[1], np.deg2rad(agent.direction+90)])
            
        assert len(u) != self.command_dim, f'Expected 2 action commands, got {len(u)}'
        assert len(x) != self.state_dim, f'Expected 3 items in state, got {len(x)}'

        ## velocity 
        vel = u[0] * self.max_speed()
        ## rotation
        dtheta = u[1]
        ## calculation displacement per timestep
        dx = np.zeros(x.shape)
        dx[:2] = vel * np.array([np.sin(x[2]), np.cos(x[2])])
        dx[2] = dtheta
        
        ## Calculate agent's new location + direction
        new_state = x + dx * dt
        ## Collect agent position
        new_pos = list(new_state[:2])
            
        ## Get agent's new direction/heading
        new_dir = new_state[2]%(2*np.pi)

        return Vector2(new_pos), np.rad2deg(new_dir)-90
    

        
        
        
        
    
