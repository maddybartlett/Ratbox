from gymnasium.envs.registration import register

########## EMPTY ENVIRONMENT ##########
## Start facing east
register(
    id="RatBox-empty-v0",
    entry_point="ratbox.envs:Simple",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"turn": 4, "agent_start_pos": (50,50), "agent_start_dir": (1,0)})

## Random start direction
register(
    id="RatBox-empty-randomDir-v0",
    entry_point="ratbox.envs:Simple",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"turn": 4, "agent_start_pos": (50,50)})

########## ENVIRONMENT WITH SINGLE WALL ##########
## Start facing east
register(
    id="RatBox-wall-v0",
    entry_point="ratbox.envs:WallRoom",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"agent_start_pos": (50,50), "agent_start_dir": (1,0)})

## Random start direction
register(
    id="RatBox-wall-randomDir-v0",
    entry_point="ratbox.envs:WallRoom",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"agent_start_pos": (50,50)})

########## TMAZE ##########
## Start facing east
register(
    id="RatBox-tmaze-v0",
    entry_point="ratbox.envs:TMazeRoom",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"agent_start_pos": (50,300), "agent_start_dir": (1,0)})

########## ENVIRONMENT FILLED WITH OBSTACLES ##########
## Start facing east
register(
    id="RatBox-blocks-v0",
    entry_point="ratbox.envs:BlockRoom",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"agent_start_pos": (50,50), "agent_start_dir": (1,0), "turn": 4})