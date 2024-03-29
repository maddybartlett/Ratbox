from ratbox.ratbox_env import RatBoxEnv, World, Goal, Agent, Wall

class TMazeRoom(RatBoxEnv):
    '''
    Maze room that is solvable by the discrete agent
    '''
    def __init__(self, width=600, height=600, speed=100000, turn=4, steering=None,
    agent_start_pos = 50, agent_start_dir=50, **kwargs):

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
            **kwargs)

    def _gen_world(self, width, height):
        # Create an empty grid
        self.world = World(width, height)

        ## Place agent in the world
        self.agent = Agent(self.agent_pos, self.agent_dir, self.speed, self.steering, self.rotation)
        self.world.add_obj(self.agent)

        ## Place goal in world
        self.goal_pos = (self.width-50,self.height-50)
        self.goal = Goal(self.goal_pos)
        self.world.add_obj(self.goal)

        ## Place walls in world
        self.world.add_obj(Wall((200,100), 400,200, name='wall_1'))
        self.world.add_obj(Wall((200,500), 400,200, name='wall_3'))
