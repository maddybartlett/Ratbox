from ratbox.ratbox_env import RatBoxEnv, World, Goal, Agent
import random

class Empty(RatBoxEnv):
    def __init__(self, width=600, height=600, speed=100000, turn=4, steering=None,
    agent_start_pos=None, agent_start_dir=None, **kwargs):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir      

        super().__init__(
            width=width,
            height=height,
            speed=speed,
            turn=turn, 
            agent_start_pos=self.agent_start_pos,
            agent_start_dir=self.agent_start_dir,
            steering=steering,
            penalty=0.01,
            **kwargs)

    def _gen_world(self, width, height):
        # Create an empty grid
        self.world = World(width, height)
        
        ## If agent start position is undefined, set random
        if self.agent_start_pos is None:
            self.agent_pos = random.sample(range(50, 500), 2)
        else:
            self.agent_pos = self.agent_start_pos

        ## If agent start direction is undefined, set random
        if self.agent_start_dir is None:
            self.agent_dir=[]
            ## Choose from a discrete set of directions for the discrete action space
            if self.steering is None or self.steering == 'discrete':
                dir_idx = random.randint(0,self.turn)
                self.agent_dir = self.StartDirs[dir_idx]
            ## Choose from infinite directions
            else:
                for i in range(2):
                    self.agent_dir.append(random.uniform(-1, 1))
        else:
            self.agent_dir = self.agent_start_dir

        ## Place agent in the world
        self.agent = Agent(self.agent_pos, self.agent_dir, self.speed, self.steering, self.rotation)
        self.world.add_obj(self.agent)

        ## Place goal in world
        self.goal_pos = (self.width-50,self.height-50)
        self.goal = Goal(self.goal_pos)
        self.world.add_obj(self.goal)
