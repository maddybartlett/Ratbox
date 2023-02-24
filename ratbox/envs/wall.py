from ratbox.ratbox_env import RatBoxEnv, World, Goal, Agent, Wall

class Wall_room(RatBoxEnv):
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
            **kwargs)

    def _gen_world(self, width, height):
        # Create an empty grid
        self.world = World(width, height)

        ## Place agent in the world
        self.agent = Agent(self.agent_pos, self.agent_dir, self.SPEED, self.steering, self.rotation)
        self.world.add_obj(self.agent)
       
        # Set goal position
        self.goal_pos = (self.width-50,self.height-50)

        self.goal = self._put_object(Goal(self.goal_pos))
        self.world.add_obj(self.goal)

        ## draw wall
        wall_pos = (250,250)
        self.wall = self._put_object(Wall(wall_pos, 50,500))
        self.world.add_obj(self.wall)

