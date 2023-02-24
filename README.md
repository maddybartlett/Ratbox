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

<pre align="center">
<img src="https://github.com/maddybartlett/Ratbox/blob/main/gifs/discrete.gif" width="400"/>
</pre>

The discrete steering models forces the state space to be discrete. The action space is limited to 3 movements:
1) turn right = 0
2) turn left = 1
3) move forward = 2

The default settings force the agent to turn 90 degrees when turning, and move forward by 100 pixels. 
The result in the discrete case is a 6x6x4 state space with 144 possible states for the agent to be in. 

The size of the available state space can be changed by changing the "turn" and "speed" arguments when initialising a new environment.
- "turn" = number of turns needed to turn 360 degrees
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

