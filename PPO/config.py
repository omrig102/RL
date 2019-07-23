from gym_environment import GymEnvironment
from mario_environment import MarioEnvironment

class Config() :

    root_dir = '.'
    episodes = 10000
    start_episode = 0
    sigma = 0.1
    theta = 0.15
    dt = 1e-2
    batch_size = 32
    buffer_size = 128
    epochs = 10
    epsilon=0.2
    entropy = 0.1
    gamma = 0.99
    lam = 0.97
    save_rate = 10
    hidden_size = 3
    hidden_units = 128
    critic_learning_rate = 0.0001
    actor_learning_rate = 0.0001
    use_conv_layers = True
    #env = MarioEnvironment('SuperMarioBros-v0',13,13,use_pixels=True,is_discrete=True,save_video_interval=5)
    env = GymEnvironment('Bowling-v0',48,48,use_pixels=True,stack_size=4,is_discrete=True,save_video=True,save_video_interval=50)
    
