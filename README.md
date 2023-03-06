# Ratbox

The Ratbox library contains a collection of optionally discrete and continuous environments to conduct research on Reinforcement Learning. 
The environments follow the Gymnasium standard API. 

## Installation
** Temporary ** <br>
To install Ratbox first clone or download this repository. Place it in a directory of your choice on your machine. Open a command prompt or terminal window and navigate to 
the Ratbox directory. <br>
e.g. `cd C:\Users\user\anaconda3\Lib\site-packages\Ratbox` <br>

Install package by running: `pip install .`

## Environments

The environments are simple 2D worlds, 600x600 pixels, that the agent (the rat) must navigate through in order to reach a goal (the cheese).

The space can be defined as either discrete or continuous by selecting the appropriate steering model.

The agent's state at any time is defined in terms of its x,y coordinate location in the world and the direction it's facing. 

## Steering Models

### Discrete

<p align="left">
<img src="https://github.com/maddybartlett/Ratbox/blob/main/gifs/discrete.gif" width="450"/>
<img src="https://github.com/maddybartlett/Ratbox/blob/main/gifs/discrete_explore.gif" width="450"/>
</p>

```
env = gym.make("RatBox-empty-v0", render_mode = "rgb_array", steering = "discrete")
``` 

The discrete steering models forces the state space to be discrete. The action space is limited to 3 movements:
1) turn right = 0
2) turn left = 1
3) move forward = 2

The default settings force the agent to turn 90 degrees when turning, and move forward by 100 pixels. 
The result in the discrete case is a 6x6x4 state space with 144 possible states for the agent to be in. 

The size of the available state space can be changed by changing the "turn" and "speed" arguments when initialising a new environment.
- "turn" = number of turns needed to complete a 360 degree rotation
- "speed" = number of pixels travelled when moving forward

For example, in `ratbox\__init__.py` you can register a new environment where the agent turns 45 degrees and can step forward 200 pixels each timestep.
```
register(
    id="RatBox-custom-v0",
    entry_point="ratbox.envs:Simple",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"turn": 8, "speed":200, "agent_start_pos": (50,50), "agent_start_dir": (1,0)})
```

### Compass

<p align="left">
<img src="https://github.com/maddybartlett/Ratbox/blob/main/gifs/compass.gif" width="450"/>
<img src="https://github.com/maddybartlett/Ratbox/blob/main/gifs/compass_explore.gif" width="450"/>
</p>

```
env = gym.make("RatBox-empty-v0", render_mode = "rgb_array", steering = "compass")
``` 

The compass steering models provides a continuous state and action space without requiring any major changes to the RL learning rule (i.e. introducing new continuous policy methods such as learning a Gaussian policy). 
We achieve this by formulating the continuous action based on a discrete set of action primitives. 

There are 4 action primitives which are stored as **direction vectors**:
1) move North = [0, -1]
2) move South = [0, 1]
3) move East = [1, 0]
4) move West = [-1, 0]

The environment takes as input a vector of 4 values describing the value of moving in each direction (these values can be generated by an RL learning rule). 
For example, the vector `[0.1,0.2,6,0.4]` would indicate that the highest value action to take is to move East. Similarly, the vector '[-2,4,4,-2]' indicates that the actions of moving South and moving East share equally high value. 

This vector can be fed into the `env.step()` function as the `action` variable. The compass steering model then performs a softmax over the vector, normalising the values to between 0 and 1, such that the sum of the vector $= 1$. 

$$ softmax([0.1,0.2,6,0.4]) = [0.003, 0.003, 0.99, 0.004] $$

$$ softmax([-2,4,4,-2]) = [0.001, 0.499, 0.499, 0.001] $$

This new normalised vector is referred to as the `weights` for the action primitives. 
The agent's movement vector - a 2-digit vector describing its motion along the x and y axes - is calculated as the weighted sum of the action primitives (the dot product weights $\cdot$ direction vectors:

$$ [0.003, 0.003, 0.99, 0.004] \cdot [North, South, East, West] = [0.986, 0] $$

$$ [0.001, 0.499, 0.499, 0.001] \cdot [North, South, East, West] = [0.498, 0.498] $$

The resultant vector describes the agent's direction of motion. As you can see, **it is not limited to the 4 action primitives but can contain any values ranging from -1 to 1. Consequently, the agent is able to move in any direction, and thereby can visit any of infinite locations within the environment**. 

A final displacement vector describing the direction *and* distance of motion is calculated by multiplying the direction vector by the agent's maximum speed of travel (default $= 100$ pixels). 

The agent's new location is then calculated by adding the displacement vector to the agent's current position. For example:

```
agent position = [50, 50]

if displacement = [98.6, 0]
agent's new position = [148.6, 50]

if displacement = [49.8, 49.8]
agent's new position = [99.8, 99.8]
```

The use of action primitives makes this steering model readily compatible with the standard, discrete formulation of Reinforcement Learning rules - rules which learn across a discrete action space. Therefore, we can use the standard, discrete action-value mapping whilst being able to operate in continuous state and action spaces. 

### Ego

<p align="center">
<img src="https://github.com/maddybartlett/Ratbox/blob/main/gifs/ego.gif" width="450"/>
</p>

```
env = gym.make("RatBox-empty-v0", render_mode = "rgb_array", steering = "ego")
``` 

The ego steering model is similar to the compass model but takes an **ego-centric** perspective. 

We still rely on the use of action primitives, but instead of using global directions as the action primitives, we use directions relative to the agent's current heading. 
1) Forward
2) Backward
3) Rightward
4) Leftward

Again the `action` given to the `env.step()` function is a vector containing 4 values, describing the value of moving in each direction. 
The first thing that the step function does is calculate the direction vectors for each action, based on the direction the agent is currently facing. 

`Forward` $=$ Agent's direction

`Backward` $=$ Agent's direction $* -1 $

`Rightward` $=$ Agent's direction rotated by $90\degree$ clockwise

`Leftward` $=$ Agent's direction rotated by $90\degree$ anti-clockwise


Once these vectors for the action primitives have been calculated, the Agent's new location and heading is calculated using the same procedure as the compass model. 

### Kinematic Unicycle

<p align="center">
<img src="https://github.com/maddybartlett/Ratbox/blob/main/gifs/unicycle.gif" width="450"/>
</p>

```
env = gym.make("RatBox-empty-v0", render_mode = "rgb_array", steering = "unicycle")
``` 

We also imnplemented the kinematic unicycle steering model. This model is commonly used as a mobile robot kinematics equation. It assumes that the vehicle is a single wheel that pivots about a central axis.   

`env.step()` takes 2 values as the `action` - **speed** and **direction** of motion. 
The **speed** value can range from -1 to 1, with negative values indicating motion in the backwards direction. 
**direction** can range between $-\pi / 2$ to $+\pi / 2$ 

### Skid-Steer

<p align="center">
<img src="https://github.com/maddybartlett/Ratbox/blob/main/gifs/skidsteer.gif" width="450"/>
</p>
