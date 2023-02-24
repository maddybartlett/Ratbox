import numpy as np 
from pygame.math import Vector2
from ratbox.utils import softmax
from .steering import SteeringModel

from gymnasium import spaces

## Constant direction vector for east
EAST = Vector2(1, 0)

class EgoModel(SteeringModel):
    '''
    Ego-centric steering, state space is (x, y, heading)
    Commands are [Forward, Backward, Rightward, Leftward]
    
    Commands are treated as action primitives and converted into a velocity (motion vector) by
    calculating a weighted sum over the probability distribution of action primitives.
    '''
    def __init__(self):
        super().__init__(state_dim=3, command_dim=4, max_speed=100)
        
        self._action_space = spaces.Box(low=np.array([0, 0, 0, 0], dtype=np.float32), 
                                        high=np.array([1, 1, 1, 1], dtype=np.float32)
                                        )
                                       

        self._observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, 0], dtype=np.float32), 
                                             high=np.array([np.inf, np.inf, 2 * np.pi], dtype=np.float32)
                                             )
        
    def action_space(self):
        return self._action_space

    def observation_space(self):
        return self._observation_space
    
    def step(self, agent, u):
        
        assert len(u) == 4, f'Expected 2 action commands, got {len(u)}'
        
        ## Action weights (i.e. the probability distribution across the action primitives)
        weights = softmax(np.asarray(u)*1)
        
        ## Agent's current direction as a vector
        dir_vec = agent.dir_vec
        
        ## Calculate direction vectors relative to agent's current direction
        Forward = dir_vec
        Backward = dir_vec*-1
        Rightward = dir_vec.rotate(90)
        Leftward = dir_vec.rotate(-90)

        self.egoDIRECTIONS = [Forward, Backward, Rightward, Leftward]
        
        ## Calculate motion vector
        move_vec = np.dot(weights, self.egoDIRECTIONS) #weighted sum
        
        ## Multiply the movement vector by the max speed to 
        ## get the final displacement vector
        displacement = move_vec * self.max_speed()
        
        ## Move the agent
        new_pos = agent.position + displacement
        
        ## New direction
        norm = np.linalg.norm(move_vec)
        if norm !=0:
            dir_vec = move_vec/norm
            dir_vec = Vector2(tuple(dir_vec))
        
        if self.max_speed() == 100:
            agent.dir_vec = dir_vec
            
        new_dir = dir_vec.angle_to(EAST)
        
        return Vector2(new_pos), new_dir


    