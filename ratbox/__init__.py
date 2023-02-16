from gymnasium.envs.registration import register

########## SIMPLE EMPTY ENVIRONMENT ##########

register(
    id="RatBox-simple-v0",
    entry_point="ratbox.envs:Simple",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"turn": 4, "agent_start_pos": (50,50), "agent_start_dir": (1,0)})

register(
    id="RatBox-simple-randomDir-v0",
    entry_point="ratbox.envs:Simple",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"turn": 4, "agent_start_pos": (50,50)})

########## ENVIRONMENT WITH SINGLE WALL ##########

register(
    id="RatBox-wall-v0",
    entry_point="ratbox.envs:Wall_room",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"turn": 4, "agent_start_pos": (50,50), "agent_start_dir": (1,0)})

register(
    id="RatBox-wall-randomDir-v0",
    entry_point="ratbox.envs:Wall_room",
    reward_threshold=95, 
    max_episode_steps=200, 
    kwargs={"turn": 4, "agent_start_pos": (50,50)})