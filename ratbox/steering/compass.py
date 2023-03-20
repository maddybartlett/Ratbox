import numpy as np
from pygame.math import Vector2
from ratbox.utils import softmax
from .steering import SteeringModel

from gymnasium import spaces

## Constant direction vectors
NORTH = Vector2(0, -1)
SOUTH = Vector2(0, 1)
EAST = Vector2(1, 0)
WEST = Vector2(-1, 0)

DIRECTIONS = [NORTH, SOUTH, EAST, WEST]

class CompassModel(SteeringModel):
    '''
    Allocentric steering, state space is (x, y, heading)
    Commands are [NORTH, SOUTH, EAST, WEST]
    Assuming a z-up coordinate system
    
    Commands are treated as action primitives and converted into a velocity (motion vector) by
    calculating a weighted sum over the probability distribution of action primitives.
    '''
    
    def __init__(self):
        self._state_dim = 3
        self._command_dim = 4
        
        super().__init__(state_dim=3, command_dim=4, max_speed=100)
        
        self._action_space = spaces.Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf], dtype=np.float32), 
                                        high=np.array([np.inf,np.inf,np.inf,np.inf], dtype=np.float32)
                                       )
                                       

        self._observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, 0], dtype=np.float32), 
                                             high=np.array([np.inf, np.inf, 2 * np.pi], dtype=np.float32)
                                             )
        
    def action_space(self):
        return self._action_space

    def observation_space(self):
        return self._observation_space
    
    def step(self, agent, u):

        assert len(u) == self._command_dim, f'Expected 4 action commands, got {len(u)}'
        
        ## Action weights
        weights = softmax(np.asarray(u)*1)
        
        
        # Calculate movement vector by calculating the dot product
        ## of the action weights and the direction vectors
        ## (dot product = the sum of the products. i.e. a
        ## weighted sum of the direction vectors)
        move_vec = np.dot(weights, DIRECTIONS) 
        
        ## Multiply the movement vector by the max speed to 
        ## get the final displacement vector
        displacement = move_vec * self.max_speed()
        
        ## Move the agent
        new_pos = agent.position + displacement
        
        ## Calculate the agent's new direction by calculating
        ## the magnitude of the movement vector
        dir_vec = agent.dir_vec
        norm = np.linalg.norm(move_vec)
        if norm !=0:
            dir_vec = move_vec/norm
            dir_vec = Vector2(tuple(dir_vec))

        if self.max_speed() == 100:
            agent.dir_vec = dir_vec
            
        new_dir = agent.dir_vec.angle_to(EAST)
        
        return Vector2(new_pos), new_dir

    