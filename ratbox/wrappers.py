from gymnasium.core import Wrapper
from gymnasium import spaces

import numpy as np

from ratbox.utils import softmax
from gymnasium.core import Wrapper

class ConvertUnicycleWrapper(Wrapper):
    '''
    Wrapper to convert from a discrete action-value policy to continuous speed/direction action space
    required by the Unicycle steering model
    
    Example
    >>> import ratbox
    >>> import gymnasium as gym
    >>> from ratbox.wrappers import UnicycleWrapper
    >>> env = gym.make("RatBox-empty-v0", render_mode = "rgb_array", steering = "unicycle")
    >>> env = ConvertUnicycleWrapper(env)
    >>> _ = env.reset()
    >>> env.steering_model.action_space()
    Box([-1.        -1.5707964], [1.        1.5707964], (2,), float32)
    >>> env.step([0,1])
    (array([100.,  50., -41., 550., 550.]), 0, False, False, {})
    
    '''
    
    def __init__(self, env):
        
        self.env = env
        
        self._action_space = spaces.Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf], dtype=np.float32), 
                                        high=np.array([np.inf,np.inf,np.inf,np.inf], dtype=np.float32)
                                       )
        
        self.env.steering_model.command_dim = 4
        self.env.steering_model.action_space = self.action_space
        
        super().__init__(env)
    
    def action_space(self):
        return self._action_space
    
    def step(self, action):
        action = self._get_action(action)
        return self.env.step(action)
    
    def _get_action(self, action):
        action = self._convert_input(self.env.agent, action)
        return action
     
    def _convert_input(self, agent, u):
        '''
        Function for converting from the 4 action logits produced by the learning rule, 
        to a velocity and rotation value needed for the Kinematic Unicycle.
        
        Inputs:
        
        agent: class
        
            the Agent object
            
        u: np.ndarray
        
            the action logits produced by the learning rule
            
        ------------
        Outputs:
        
        np.ndarray [velocity, rotation]
            
            an array containing a value ranging [-1, 1] describing the agent's velocity, and a value ranging [0, 360] describing 
            the agent's angle of rotation.    
        
        '''
        
        ## check action and state have correct number of elements
        assert len(u) == 4, f'Expected 4 action commands, got {len(u)}'

        ## Action weights
        u = np.asarray([u[:2], u[2:]])
        weights = np.array([softmax(u[0]*1), softmax(u[1]*1)])
            
        ## Agent's direction as a vector
        dir_vec = agent.dir_vec
            
        ## convert the incoming action distribution into velocity and rotation
        accelerate = dir_vec
        brake = dir_vec*0
        
        VEL_PRIMITIVES = [accelerate, brake]
        
        ## get weighted vectors for each action
        ## this rescales velocity between -1 and 1
        velocity = np.dot(weights[0], VEL_PRIMITIVES)
        
        # get single value for velocity
        v = np.sum(velocity)
        
        ## rotation 
        rotation_vec = np.diff(weights[1])
        theta = int(90 * rotation_vec)
        
        return np.array([v, np.deg2rad(theta)])  
    
Full_Forward = [1,1]
Left_Turn = [0,1]
Right_Turn=[1,0]

SKID_ACTIONS = [Full_Forward, Left_Turn, Right_Turn]
    
class ConvertSkidWrapper(Wrapper):
    '''
    Wrapper to convert from a discrete action-value policy to continuous speed/direction action space
    required by the Skid-Steer steering model
    
    Example
    >>> import ratbox
    >>> import gymnasium as gym
    >>> from ratbox.wrappers import UnicycleWrapper
    >>> env = gym.make("RatBox-empty-v0", render_mode = "rgb_array", steering = "unicycle")
    >>> env = ConvertSkidWrapper(env)
    >>> _ = env.reset()
    >>> env.steering_model.action_space()
    Box(0.0, 1.0, (4,), float32)
    >>> env.step([0,1])
    (array([1.00000000e+02, 5.00000000e+01, 4.03491405e-01, 5.50000000e+02,
        5.50000000e+02]),
    0,
    False,
    False,
    {})
    
    '''
    
    def __init__(self, env):
        
        self.env = env
    
    def step(self, action):
        action = self._get_action(action)
        return self.env.step(action)
    
    def _get_action(self, action):
        action = self._convert_input(self.env.agent, action)
        return action
     
    def _convert_input(self, agent, u):
        '''
        Function for converting from the 4 action logits produced by the learning rule
        to the speeds of the left and right tracks for the Skid-Steer steering model
        (assuming the agent is a tank-like vehicle).
        
        This is done by treating the 4 action logits as though they correspond with
        4 directions (forward, backward, rightward, leftward). 
        A softmax is applied to get the weights for each direciton. 
        The speed for the right track is calculated as the sum of the weights for the
        forward, backward and rightward directions. Similarly, the speed for the left
        track is the sum of the weights for the forward, backward and leftward directions.
        
        The resultant values are normalised so that they lie between -1 and 1.  
        
        Inputs:
        
        agent: class
        
            the Agent object
            
        u: np.ndarray
        
            the action logits produced by the learning rule
            
        ------------
        Outputs:
        
        np.ndarray [right track speed, left track speed]
        
        an array containing a value ranging [-1, 1] describing the speed of each track
        
        '''
        ## check action and state have correct number of elements
        assert len(u) == 3, f'Expected 4 action commands, got {len(u)}'
        
        ## Action weights
        weights = softmax(np.asarray(u)*1)
        
        ## Choose an action
        action = SKID_ACTIONS[weights.argmax()]
        
        return action
    