# Ratbox

The Ratbox library contains a collection of optionally discrete and continuous environments to conduct research on Reinforcement Learning. 
The environments follow the Gymnasium standard API. 

## Installation
** Temporary ** <br>
To install Ratbox first clone or download this repository. Place it in a directory of your choice on your machine. Open a command prompt or terminal window and navigate to 
the ratbox directory. <br>
e.g. `cd C:\Users\user\anaconda3\Lib\site-packages\ratbox` <br>

Install package by running: `pip install --user -e .`

Whenever you import this package you will need to add the path to the sys.path. 
To do this, just add the following lines (swapping in your path to the ratbox directory) to the beginning of your script. 

```
import sys
sys.path.append('C:\\Users\\Path\\To\\Ratbox')
```

## Environments

The environments are simple 2D worlds that the agent (the rat) must navigate through in order to reach a goal (the cheese).

The space can be defined as either discrete or continuous by selecting the appropriate steering model.

## Steering Models

### Discrete

The discrete steering models forces the state space to be discrete. The action space is limited to 3 movements:
1) 0 = turn right
2) 1 = turn left
3) 2 = move forward

The default settings force the agent to turn 90 degrees when turning, and move forward by 100 pixels. This can be changed by changing the "turn" and "speed" arguments when initialising a new environment.
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

