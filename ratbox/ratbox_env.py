import gymnasium as gym
from gymnasium import spaces 

import numpy as np
from enum import IntEnum
from typing import Optional

import pygame
from pygame.math import Vector2
from pygame.image import load
from pygame.transform import rotozoom

from utils import load_sprite, softmax
from steering import SteeringModel, DiscreteModel, KinematicUnicycle, CompassModel, EgoModel, SkidSteer


# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "forest_green": np.array([34,139,34]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "black": np.array([0, 0, 0])
}

# Map of steering model strings to model classes
STEERING = {
    "discrete": DiscreteModel(),
    "compass": CompassModel(),
    "ego": EgoModel(),
    "unicycle": KinematicUnicycle(),
    "skidsteer": SkidSteer(),
}

## Constant direction vectors
NORTH = Vector2(0, -1)
SOUTH = Vector2(0, 1)
EAST = Vector2(1, 0)
WEST = Vector2(-1, 0)

DIRECTIONS = [NORTH, SOUTH, EAST, WEST]

## Create game object
class WorldObject:
    '''
    Parent class for objects placed in the environment/world
    e.g. goal and obstacles like walls
    Does not include agent object
    '''
    def __init__(self, position, name=None): 
        ## Initialise position and sprite name (i.e. filename for png image)
        self.position = Vector2(position)
        self.name = name

    def draw(self, surface):
        ## Load sprite and retrieve radius. Do not draw to surface
        self.sprite = load_sprite(self.name)
        self.radius = self.sprite.get_width()/2

class Goal(WorldObject):
    '''
    Goal object 
    Stationary
    '''
    def __init__(self, position, name='cheese'):
        self.name = name
        self.position = position
        self.dir_vec = Vector2(1,0)
        self.direction = self.dir_vec.angle_to(EAST) #in degrees

        self.can_overlap = True

class Wall(WorldObject):
    '''
    Stationary w x h pixel obstacle in the environment, through which the agent cannot pass
    '''
    def __init__(self, position, w=10, h=10, name='wall'):
        self.position = position # x,y location of the centre of the wall
        self.sprite=None # set to None to distinguish from Goal and Agent
        self.w = w # wall width
        self.h = h # wall height

        self.can_overlap = False

        self.name = name

        ## Create the wall object
        self.rect = pygame.Rect(self.position[0],self.position[1],self.w, self.h)
        ## Place wall in world
        self.rect.center = self.position

    def draw(self, surface):
        color = COLORS["grey"]
        pygame.draw.rect(surface, color, self.rect)


class Agent():
    '''
    Mobile Agent object 
    '''
    def __init__(self, position, facing, speed, steering, rotation=None, name="rat"):
        self.position = Vector2(position)
        self.steering = steering
        ## Initialise agent as facing east
        self.dir_vec = Vector2(facing)
        ## Get current direction relative to East
        self.direction = self.dir_vec.angle_to(EAST) #in degrees

        self.discrete_rotation = rotation #degrees per turn when using discrete action space

        ## record whether agent has bumped into something
        self.collision = False 
        
        ## set steering model (default is discrete)
        if self.steering is None:
            self.steering = 'discrete'  
        self.travel = STEERING[self.steering]      
        
        self.travel._max_speed = speed #max pixels per step

        self.name = name

    def draw(self, surface):
        self.sprite = load_sprite(self.name)
        self.radius = self.sprite.get_width()/2


class World:
    ''' 
    Create the world the agent will move around in
    '''

    def __init__(self, width, height):
        ## set minimum size of box (in pixels)
        assert width >= 100
        assert height >= 100

        self.width = width
        self.height = height

        self.contents = {}

    def add_obj(self, object):
        #####
        self.contents[object.name] = object       

    

class RatBoxEnv(gym.Env):
    """
    2D environment
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
        "steering_models": ["discrete", "compass", "ego", "unicycle", "skidsteer"]
    }

    class Discrete_Actions(IntEnum):
        # move forward, turn right, turn left 
        forward = 0
        right = 1
        left = 2

    class Compass_Actions(IntEnum):
        # move north, south, east or west
        north = 0
        south = 1
        east = 2
        west = 3

    class Ego_Actions(IntEnum):
        # move forward, backward, rightward or leftward
        forward = 0
        backward = 1
        rightward = 2
        leftward = 3

    def __init__(self, 
                width: int = 600,
                height: int = 600,
                speed: int = 100, 
                turn: int = None,
                agent_start_pos: list = None,
                agent_start_dir: int = None,
                max_steps: int = 500,
                steering: Optional[str] = None,
                render_mode: Optional[str] = None):
        

        ## Environment configuration
        self.width = width
        self.height = height
        self.max_steps=max_steps ##max number of time steps per trial

        ## Initialise render window as None
        self.window=None

        ## Starting position and direction of the agent
        self.agent_pos = agent_start_pos
        self.agent_dir = agent_start_dir

        ## Agent mobility
        self.turn=turn ##number of times it can turn in a circle (only dor discrete actions)
        self.steering = steering ##steering model string id
        self.speed = speed
        
        if self.turn is not None:
            self.rotation = 360/self.turn ##degrees rotated per timestep when using discrete action space

            ## Convert to vectors to generate possible start directions
            self.StartDirs = [[1,0]]
            for h in range(self.turn):
                nxt = Vector2(self.StartDirs[-1]).rotate(self.rotation)[:]
                self.StartDirs.append(nxt)

        ## rendering attributes
        self.render_mode = render_mode

        ## steering model (default is discrete)
        if self.steering is None:
            self.steering = 'discrete'   
        self.steering_model = STEERING[self.steering]

        self.action_space = self.steering_model.action_space()
        self.n_actions = self.action_space.shape[0]
        self.observation_space = spaces.Box(low=-1000, high=1000,
                                            shape=(5,), dtype=np.float64)

        ## Current world
        self.world = World(self.width, self.height)


    def reset(self, seed=None, options=None):
        '''Reset the environment at the beginning of each learning trial'''
        self.done = False
        self.max_rew = 1

        # Generate a new random grid at the start of each episode
        self._gen_world(self.width, self.height)

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        return obs, {}

    def _put_object(self, obj: WorldObject):
        '''Put an object in a specific position in the box'''
        return obj

    def _get_game_objects(self):
        '''Fetch a list of objects in the environment'''
        game_object=[]
        for obj in self.world.contents:
            game_object.append(self.world.contents[obj])

        return game_object


    def _gen_world(self, width, height):
        '''generate the world as defined in the specific environment class (e.g. Simple, Wall_room)'''
        pass

    def gen_obs(self):
        '''Return the agent's position and direction and the location of the goal'''
        
        agent_x = self.agent.position[0]
        agent_y = self.agent.position[1]
        direction = self.agent.direction
        goal_x = self.goal.position[0]
        goal_y = self.goal.position[1]

        obs = [agent_x, agent_y, direction, goal_x, goal_y]
        obs = np.array(obs)

        return obs

    def step(self, action):
        '''Move agent/world forward one time step'''
        self.step_count += 1 ## increase step count by 1

        self.reward = 0 
            
        ## Move the agent
        new_pos, new_dir = self.agent.travel.step(self.agent,
                                                  action)

        ## check for collisions and adjust accordingly
        self.agent.position = self._check_collision(new_pos, action)
        self.agent.direction = new_dir
   
        ## Penalty for bumping into walls/obstacles
        if self.agent.collision == True:
            self.reward = -0.01
            self.agent.collision = False

        ## Reward and done when collide with goal
        dist = np.abs(self.agent.position-self.goal.position)
        if dist[0] < 50 and dist[1] < 50:
            ## Reward discounted by number of steps it took to reach the goal
            self.reward = self.max_rew - 0.9 * (self.step_count/self.max_steps)
            self.done = True

        ## Create observation
        ## observation = rat_x, rat_y, direction, cheese_x, cheese_y
        agent_x = self.agent.position[0]
        agent_y = self.agent.position[1]
        direction = self.agent.direction
        goal_x = self.goal.position[0]
        goal_y = self.goal.position[1]

        self.observation = [agent_x, agent_y, direction, goal_x, goal_y]
        self.observation = np.array(self.observation)

        info = {}
        return self.observation, self.reward, self.done, False, info
    
    def _check_collision(self, agent_pos, action):
        '''Check the agent's path for collisions. '''
        
        pos_in_bounds = self._check_bounds(agent_pos)
        new_pos = self._check_obstacles(pos_in_bounds, action)
        
        return new_pos
    
    def _check_bounds(self, agent_pos):
        '''
        Check whether the agent's path takes it outside world bounds. 
        If so, stop the agent at the world bound.
        '''
        ## World bounds are 50 pixels in from the edges of the rendered surface
        width = self.world.width - 50
        height = self.world.height - 50

        ## Check whether new position is within world bounds.
        ## If agent is outside bounds, move back and set collision to True
        if agent_pos[0] > width:
            agent_pos[0] = width
            self.agent.collision = True
        if agent_pos[0] < 50:
            agent_pos[0] = 50
            self.agent.collision = True
        if agent_pos[1] > height: 
            agent_pos[1] = height
            self.agent.collision = True
        if agent_pos[1] < 50:
            agent_pos[1] = 50
            self.agent.collision = True

        return agent_pos

        
    def _check_obstacles(self, agent_pos, action):
        '''
        Check if the agent's path crosses over a solid object. 
        If so, stop the agent once it reaches the object.
        
        Current the wall is the only solid object. 
        This function will need to change when more objects are added
        '''
        new_pos = agent_pos
        
        if 'wall' in self.world.contents:
            ## Get the location of the wall by drawing it
            wall = self.world.contents['wall'].rect
            pygame.display.init()
            pygame.display.set_mode((self.world.width, self.world.height), flags = pygame.HIDDEN)
            
            ## Take into account the radius of the agent sprite
            sprite = load_sprite(self.agent.name)
            radius = sprite.get_width()/2

            ## Get 10 locations along agent's path, by repeating the action at 10 different speeds
            distance = ((self.agent.position[0] - new_pos[0])**2 + (self.agent.position[1] - new_pos[1])**2)**0.5  
            speeds = np.linspace(0, distance, 10) 

            trajectory=[]
            for speed in speeds:
                self.agent.travel._max_speed = speed
                trajectory.append(self.agent.travel.step(self.agent, action))
                
            self.agent.travel._max_speed = 100
                
            ## will the agent cross the wall?
            for i in range(len(trajectory)):
                ## Turn the trajectory location into a rect object on the world surface
                ## with the same radius as the agent
                agent_rect = pygame.Rect(trajectory[i][0][0], trajectory[i][0][1], radius, radius)
                
                ## Check if agent collided with wall
                if pygame.Rect.colliderect(wall, agent_rect):
                    ## register the collision and penalise for it
                    self.agent.collision=True
                    ## if the agent will immediately cross the wall, 
                    ## just rotate the agent, don't move forward
                    if len(trajectory[:i])==0:
                        new_pos = self.agent.position
                    ## otherwise, move the agent to the last possible position
                    new_pos = trajectory[i-1][0]
                    break
                    
        return new_pos     

    def render(self):
        ''' 
        Render the current state of the world as an image using pygame.
        
        If render_mode = human, image will be rendered as a pop-up window on screen.
        If render_mode = rgb_array, image will be an array of values which can be 
        viewed using matplotlib's imshow. 
        '''
        ## call pygame
        pygame.init()

        ## Screen dimensions in pixels
        self.screen_width = 600
        self.screen_height = 600

        ## for "human" mode, show the render window on screen
        if self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height), flags = pygame.SHOWN
            )
        elif self.render_mode == "rgb_array":
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height), flags = pygame.HIDDEN
            )

        self.clock = pygame.time.Clock() ## use clock

        ## Set screen background
        self.background = pygame.Surface((self.screen_width, self.screen_height))
        self.background.fill(COLORS["forest_green"])
        self.window.blit(self.background, (0, 0))

        ## fetch game objects
        game_object = self._get_game_objects()
    
        for i in range(len(game_object)):
            obj = game_object[i]
            ## draw object
            obj.draw(self.window)

            ## if there's a sprite, rotate and calculate size and position of sprite
            if obj.sprite is not None:
                self.angle = obj.direction
                rotated_obj = rotozoom(obj.sprite, self.angle, 1.0)
                rotated_obj_size = Vector2(rotated_obj.get_size())

                ## calculate position
                blit_position = obj.position - rotated_obj_size * 0.5

                self.window.blit(rotated_obj, blit_position)

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            img = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
                )

            return img 

    def close(self):
        '''Close the render window.'''
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False