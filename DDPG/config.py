from gym_environment import GymEnvironment

class Config() :

    episodes = 10000
    steps = 7000
    memory_size = 1000000
    initial_memory_size = 0
    update_rate = 1
    target_update_rate = 1
    batch_size = 100
    sigma = 0.1
    theta = 0.15
    dt = 1e-2
    noise_scale = 0.1
    render = False
    save_rate = 1
    root_dir = '.'
    gamma = 0.99
    epsilon = 0.1
    epsilon_decay = 100000
    learning_rate_actor = 0.0001
    learning_rate_critic = 0.001
    TAU = 0.001
    env = GymEnvironment('BipedalWalker-v2',13,13,use_pixels=False)