import numpy as np
from pygame.math import Vector2
from ratbox.utils import softmax
from .steering import SteeringModel

from gymnasium import spaces

## Constant direction vector
EAST = Vector2(1, 0)

class DiscreteModel(SteeringModel):
    '''
    Discrete steering, state space is (x, y, heading)
    Commands are [turn right, turn left, move forward]
    '''
    
    def __init__(self):
        super().__init__(state_dim=3, command_dim=4, max_speed=100)
        
        self._action_space = spaces.Box(low=np.array([0, 0, 0], dtype=np.float32), 
                                        high=np.array([np.inf, np.inf, np.inf], dtype=np.float32)
                                        )
                                       

        self._observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, 0], dtype=np.float32), 
                                             high=np.array([np.inf, np.inf, 2 * np.pi], dtype=np.float32)
                                             )
        
    def action_space(self):
        return self._action_space

    def observation_space(self):
        return self._observation_space
    
    def step(self, agent, u):
        
        assert len(u) == 3, f'Expected 3 action commands, got {len(u)}'
        
        ## Clip actions to be within action space bounds
        for i in range(len(u)):
            u[i] = np.clip(u[i], 0, np.inf).astype(np.float32)
        
        ## Action weights
        weights = softmax(np.asarray(u)*1)
        
        ## Get index of max action weight
        action = weights.argmax()

        ## Rotate right
        if action == 0:
            angle = agent.discrete_rotation * 1
            dir_vec = agent.dir_vec.rotate(angle)
            new_pos = agent.position           
        ## Rotate left
        elif action == 1:
            angle = agent.discrete_rotation * -1
            dir_vec = agent.dir_vec.rotate(angle)
            new_pos = agent.position
        ## Move forward
        elif action == 2:
            displacement = agent.dir_vec * self.max_speed()
            
            ## Move the agent
            new_pos = agent.position + displacement

            dir_vec = agent.dir_vec
        
        if self.max_speed() == 100:
            agent.dir_vec = dir_vec
            
        new_dir = dir_vec.angle_to(EAST)

        return Vector2(new_pos), new_dir