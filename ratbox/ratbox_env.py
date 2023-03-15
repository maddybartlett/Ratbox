import gymnasium as gym
from gymnasium import spaces 

import numpy as np
import math
from enum import IntEnum
from typing import Optional

import pygame
from pygame.math import Vector2
from pygame.image import load
from pygame.transform import rotozoom

from ratbox.steering import SteeringModel, DiscreteModel, KinematicUnicycle, CompassModel, EgoModel, SkidSteer
from ratbox.utils import load_sprite, softmax


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
#class WorldObject:
#    '''
#    Parent class for objects placed in the environment/world
#    e.g. goal and obstacles like walls
#    Does not include agent object
#    '''
#    def __init__(self, position, name=None): 
        ## Initialise position and sprite name (i.e. filename for png image)
#        self.position = Vector2(position)
#        self.name = name

#    def draw(self, surface):
        ## Load sprite and retrieve radius. Do not draw to surface
#        self.image = load_sprite(self.name)
#        self.radius = self.image.get_width()/2

class Goal():
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
        
    def draw(self, surface):
        ## Load sprite and retrieve radius. Do not draw to surface
        self.image = load_sprite(self.name)
        self.radius = self.image.get_width()/2
        
class Wall():
    '''
    Stationary w x h pixel obstacle in the environment, through which the agent cannot pass
    '''
    def __init__(self, position, w=10, h=10, name='wall', degrees=0):
        self.position = position # x,y location of the centre of the wall
        self.image=None # set to None to distinguish from Goal and Agent
        self.w = w # wall width
        self.h = h # wall height

        self.can_overlap = False

        self.name = name

        ## Create the wall object
        wall = pygame.Surface([self.w, self.h])
        wall.fill(COLORS["grey"])
        
        ## Rotate wall
        rotatedWall = pygame.transform.rotate(wall, degrees)
        self.rect = rotatedWall.get_rect(center = position)

    def draw(self, surface):
        color = COLORS["grey"]
        pygame.draw.rect(surface, color, self.rect)       
        
class Ball():
    '''
    Stationary circle object in the environment which the agent can't pass through.
    '''     
    def __init__(self, position, radius=10, colorName="red", name="ball"):
        self.position = position
        self.radius = radius
        self.color = COLORS[colorName]
        self.name = name
        
    def draw(self, surface):
        #pygame.draw.circle(surface, self.color, )
        pass
        

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
        self.image = load_sprite(self.name)
        self.radius = self.image.get_width()/2


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
        self.max_steps = max_steps ##max number of time steps per trial

        ## Initialise render window as None
        self.window = None

        ## Starting position and direction of the agent
        self.agent_pos = agent_start_pos
        self.agent_dir = agent_start_dir

        ## Agent mobility
        self.turn = turn ##number of times it can turn in a circle (only dor discrete actions)
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
        
        ## Make sure steering model is valid     
        try:
            self.steering_model = STEERING[self.steering]
        except KeyError:
            keysList = list(STEERING.keys())
            keysString = ', '.join(keysList)
            raise(KeyError(f'Invalid steering model entered. Please use one of the following: {keysString}'))

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
        obs = self._gen_obs()

        return obs, {}

    def _get_game_objects(self):
        '''Fetch a list of objects in the environment'''
        game_object=[]
        for obj in self.world.contents:
            game_object.append(self.world.contents[obj])

        return game_object


    def _gen_world(self, width, height):
        '''generate the world as defined in the specific environment class (e.g. Simple, Wall_room)'''
        pass

    def _gen_obs(self):
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
        self.old_dir_vec = self.agent.dir_vec
        new_pos, new_dir = self.agent.travel.step(self.agent, action)

        ## check for collisions and adjust accordingly
        self.agent.position, self.agent.direction = self._check_collision(new_pos, new_dir, action)

        ## Penalty for bumping into walls/obstacles
        if self.agent.collision == True:
            self.reward = -0.1
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
    
    def _check_collision(self, agent_pos, agent_dir, action):
        '''Check the agent's path for collisions. '''
        new_dir = agent_dir
        
        ## Get list of obstacles names
        object_names = list(self.world.contents.keys())[2:]
        
        ## If there are obstacles in the world
        if len(object_names) != 0:
            ## Add them to a list of rect objects
            object_list = []
            
            for name in object_names:
                object_list.append(self.world.contents[str(name)].rect)
            
            new_dir = agent_dir   
            #new_dir = self._check_turn(object_list, agent_dir)
            
            ## convert direction to Radians
            #rad = np.deg2rad(new_dir)
            ## convert to vector
            #Vx = np.cos(rad)
            #Vy = np.sin(rad)
            #vec = (Vx, Vy)
            #self.agent.dir_vec = Vector2(vec)
            
            new_pos = self._check_speed(object_list, agent_pos, new_dir, action)
            
            pos_in_bounds = self._check_bounds(new_pos)

        return pos_in_bounds, new_dir
    
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

    def _check_turn(self, object_list, new_dir):

        ## Get agent's starting direction and position
        old_dir = self.agent.direction
        old_pos = self.agent.position
                    
        ## Create an agent sprite for collision checking 
        agent_sprite = load_sprite(self.agent.name)
        
        ## Create a list of angles between the agent's starting heading and target heading            
        angles = np.linspace(old_dir, new_dir, 5) 
        
        ## Variable for retrieving the index of the last viable angle
        angle_index = -1
        
        for deg_i in range(len(angles)):
            ## Turn the agent sprite by each angle
            rotated_agent = pygame.transform.rotate(agent_sprite, angles[deg_i])
            
            ## Place the agent sprite in the world
            agent_rect = rotated_agent.get_rect(center=(old_pos[0],old_pos[1]))
            
            ## Check for collisions
            index = pygame.Rect.collidelist(agent_rect, object_list) 
            if index != -1:
                angle_index = deg_i-1
                break
        
        direction = angles[angle_index]
        
        return direction
    
           
    def _check_speed(self, object_list, new_pos, new_dir, action):
        
        ## Get agent's starting position
        old_pos = self.agent.position
        
        ## Create an agent sprite for collision checking 
        agent_sprite = load_sprite(self.agent.name)
        ## Turn the agent sprite by the selected angle
        rotated_agent = pygame.transform.rotate(agent_sprite, new_dir)
        
        ## Calculate distance from start to target location
        distance = ((old_pos[0] - new_pos[0])**2 + (old_pos[1] - new_pos[1])**2)**0.5
        ## Create a list of speeds between 0 and speed needed to reach target location
        speeds = np.linspace(0, distance, 5) 
        
        ## Create a list of locations between start and target
        locations=[]
        for speed in speeds:
            self.agent.travel._max_speed = speed
            locations.append(self.agent.travel.step(self.agent, action))
                
        ## Reset the agent's speed 
        self.agent.travel._max_speed = 100
        
        ## Variable for retrieving the index of the last viable angle
        location_index = -1
        for pos_i in range(len(locations)):
            ## Place the agent sprite in the world
            agent_rect = rotated_agent.get_rect(center=(locations[pos_i][0][0],locations[pos_i][0][1]))
            
            ## Check for collisions
            index = pygame.Rect.collidelist(agent_rect, object_list) 
            if index != -1:
                if pos_i > 0:
                    location_index = pos_i-1
                else: 
                    location_index = 0
                break
            
        position = locations[location_index][0]
        
        return position
        
        
    def _check_obstacles(self, agent_pos, agent_dir, action):
        '''
        Check if the agent's path crosses over a solid object. 
        If so, stop the agent once it reaches the object.
        '''
        new_pos = agent_pos
        new_dir = agent_dir
        old_dir = self.agent.direction
                
        ## If there are objects in the world, check for collisions
        if len(self.world.contents) > 2:
            pygame.display.init()
            pygame.display.set_mode((self.world.width, self.world.height), flags = pygame.HIDDEN)
            
            ## Take into account the radius of the agent sprite
            sprite = load_sprite(self.agent.name)
            rotated_sprite = pygame.transform.rotate(sprite, new_dir)
            
            ## Check if the agent's step size needs to be shorter
            ## Get 10 locations along agent's path, by repeating the action at 10 different speeds
            distance = ((self.agent.position[0] - new_pos[0])**2 + (self.agent.position[1] - new_pos[1])**2)**0.5  
            speeds = np.linspace(0, distance, 10) 

            trajectory=[]
            for speed in speeds:
                self.agent.travel._max_speed = speed
                trajectory.append(self.agent.travel.step(self.agent, action))
                
            self.agent.travel._max_speed = 100
                
            for obj in list(self.world.contents.keys())[2:]:
                obj_rect = self.world.contents[str(obj)].rect
                
                for i in range(len(trajectory)):
                    ## Turn the trajectory location into a rect object on the world surface
                    ## with the same radius as the agent
                    agent_rect = rotated_sprite.get_rect(center = (trajectory[i][0][0], trajectory[i][0][1]))  
                    
                    if self.agent.steering == 'discrete' and i==0 and pygame.Rect.colliderect(obj_rect, agent_rect):
                        break
                    
                    ## Check if agent collided with wall
                    if pygame.Rect.colliderect(obj_rect, agent_rect):
                        ## register the collision and penalise for it
                        self.agent.collision=True
                        ## if the agent will immediately cross the wall, 
                        ## just rotate the agent, don't move forward
                        if len(trajectory[:i])==0 or self.agent.steering == 'discrete':
                            new_pos = self.agent.position
                            if self.agent.steering == 'discrete':
                                self.agent.dir_vec = self.old_dir_vec
                                new_dir = self.agent.direction
                        ## otherwise, move the agent to the last possible position
                        else:
                            new_pos = trajectory[i-1][0]
                        break      
            
            ## Check if the agent's rotation needs to be reduced
            for obj in list(self.world.contents.keys())[2:]:
                obj_rect = self.world.contents[str(obj)].rect
                if self.agent.steering != 'discrete':            
                    if  old_dir != new_dir:
                        angles = np.linspace(old_dir, new_dir, 10) 
                        angles = sorted(angles, key=abs)
                        
                        for i in range(len(angles)):
                            sprite = load_sprite(self.agent.name)
                            rotate_sprite = pygame.transform.rotate(sprite, angles[i])
                            
                            agent_rect = rotate_sprite.get_rect(center = (new_pos[0], new_pos[1]))
                                
                            if pygame.Rect.colliderect(obj_rect, agent_rect):
                                ## register the collision and penalise for it
                                self.agent.collision=True
                                if len(angles[:i])==0:
                                    new_dir = self.agent.direction
                                else:
                                    new_dir = angles[i-1]
                                break
                            break
       
        ## convert direction to Radians
        rad = np.deg2rad(new_dir)
        ## convert to vector
        Vx = np.cos(rad)
        Vy = np.sin(rad)
        vec = (Vx, Vy)
        self.agent.dir_vec = Vector2(vec)
          
        return new_pos, new_dir      

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
        self.screen_width = self.width
        self.screen_height = self.height

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
            if obj.image is not None:
                self.angle = obj.direction
                rotated_obj = rotozoom(obj.image, self.angle, 1.0)
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