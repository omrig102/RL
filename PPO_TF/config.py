from gym_environment import GymEnvironment
from mario_environment import MarioEnvironment

class Config() :

    #game = 'SuperMarioBros-v0'
    game = 'BipedalWalker-v2'
    root_dir = '.'
    episodes = 10000
    start_episode = 0
    batch_size = 64
    buffer_size = 8192
    epochs = 10
    epsilon=0.2
    entropy = 0.1
    gamma = 0.99
    l2 = 0.001
    save_rate = 10
    critic_learning_rate = 0.0001
    actor_learning_rate = 0.0001
    use_shuffle = True
    use_pixels = False
    if(use_pixels) :
        resized_height = 13
        resized_width = 13
    network_type = 'mlp'
    #mlp
    hidden_layers = 3
    hidden_units = 128
    #lstm
    if(network_type == 'lstm') :
        lstm_layers = 1
        lstm_units = 128
        timestamps = 4
    #conv2d
    elif(network_type == 'conv2d' and use_pixels) :
        conv_layers = 2
        conv_units = 128
        stack_size = 4
        

    
    
    
    
    save_video = True
    save_video_interval = 10
    #env = MarioEnvironment(game,save_video=save_video,save_video_interval=save_video_interval)
    env = GymEnvironment(game,save_video=save_video,save_video_interval=save_video_interval)
