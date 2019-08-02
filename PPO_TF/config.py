from gym_environment import GymEnvironment
from mario_environment import MarioEnvironment

class Config() :

    game = 'LunarLanderContinuous-v2'
    root_dir = '.'
    episodes = 10000
    start_episode = 0
    sigma = 0.1
    theta = 0.15
    dt = 1e-2
    batch_size = 64
    buffer_size = 8192
    epochs = 10
    epsilon=0.2
    entropy = 0.1
    gamma = 0.99
    l2 = 0.001
    TAU = 1
    save_rate = 10
    noise_interval = 128
    hidden_size = 3
    hidden_units = 128
    critic_learning_rate = 0.0001
    actor_learning_rate = 0.0001
    use_conv_layers = False
    use_lstm_layers = True
    use_shuffle = False
    resized_height = 13
    resized_width = 13
    use_pixels = False
    stack_size = 4
    is_discrete = False
    save_video = True
    save_video_interval = 10
    #env = MarioEnvironment('SuperMarioBros-v0',13,13,use_pixels=True,stack_size=4,is_discrete=True,save_video_interval=5)
    env = GymEnvironment(game,resized_height,resized_width,use_pixels=use_pixels,stack_size=stack_size
    ,is_discrete=is_discrete,save_video=save_video,save_video_interval=save_video_interval)
