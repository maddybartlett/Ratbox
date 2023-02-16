import numpy as np


class SteeringModel(object):
    def __init__(self, state_dim : int = 1, command_dim : int = 1, max_speed : int = 1):
        self.state = np.zeros((state_dim,))
        self._command_dim = command_dim
        self._max_speed = max_speed
        pass

    def command_dim(self):
        return self._command_dim

    def state_dim(self):
        return self.state.shape
    
    def max_speed(self):
        return self._max_speed

    def action_space(self):
        raise NotImplementedError('Subclass me!')

    def step(x, u, dt=1):
        '''
        Computes update to agent state assuming simple Euler integration.

        parameters:
        -----------
        x : np.ndarray

            The current state of the agent

        u : np.ndarray

            The command sent to the agent

        dt : float

            The time step we are simulating for.
        '''
        raise NotImplementedError('Subclass me!')




