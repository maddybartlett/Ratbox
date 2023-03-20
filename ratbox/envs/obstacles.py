from ratbox.ratbox_env import RatBoxEnv, World, Goal, Agent, Wall, Ball

class BlockRoom(RatBoxEnv):
    '''
    A room full of square/rectangular blocks the agent has to move around
    '''
    def __init__(self, width=600, height=600, speed=100, turn=4, steering=None,
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
            max_reward=100,
            penalty = 0.5,
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

        ## Place blocks in world
        self.world.add_obj(Wall((225,100), 60,60, colorName="yellow", name='block_1', degrees=45))
        self.world.add_obj(Wall((125,500), 50,50, colorName="yellow", name='block_4', degrees = -75))
        #self.world.add_obj(Wall((250,250), 50,50, colorName="yellow", name='block_middle'))
        
        ## Place balls in the world
        self.world.add_obj(Ball((300,550), radius=37.5, colorName="blue", name="ball_3"))
        self.world.add_obj(Ball((362.5,312.5), radius=37.5, colorName="blue", name='ball_4'))