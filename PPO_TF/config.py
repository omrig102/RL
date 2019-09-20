from gym_environment import GymEnvironment
from mario_environment import MarioEnvironment
from retro_environment import RetroEnvironment
import pickle


class Config() :

    #game = 'SuperMarioBros-v0'
    game = 'BipedalWalker-v2'
    root_dir = '.'
    log_dir = 'logs'
    with_summary = True
    timesteps = 2e7
    start_timestep = 0
    save_video = True
    save_video_interval = 10
    save_rate = 10
    log_episodes = 10
    batch_size = 64
    buffer_size = 2048
    epochs = 10
    gradient_clip = 0.5
    epsilon=0.2
    min_epsilon = 0.2
    entropy = 0.0
    gae = 0.95
    gamma = 0.99
    reward_scaler = 'sign'
    actor_critic_type = 'seperate'
    
    critic_learning_rate = lambda f: 3e-4 * f
    actor_learning_rate = lambda f: 3e-4 * f
    use_shuffle = True
    use_pixels = False
    if(use_pixels) :
        resized_height = 48
        resized_width = 48
    network_type = 'mlp'
    #mlp
    mlp_hidden_layers = 1
    mlp_hidden_units = [512]
    #lstm
    if(network_type == 'lstm') :
        lstm_layers = 1
        lstm_units = 128
        unit_type = 'gru'
        timestamps = 4
    #conv2d
    elif(network_type == 'conv2d' and use_pixels) :
        conv_layers = 3
        conv_units = [32,64,64]
        strides = [[4,4],[2,2],[1,1]]
        kernel_size = [8,4,3]
        stack_size = 4
    
    #env = MarioEnvironment(game,save_video=save_video,save_video_interval=save_video_interval)
    #env = RetroEnvironment(game,save_video=save_video,save_video_interval=save_video_interval)
    env = GymEnvironment(game,save_video=save_video,save_video_interval=save_video_interval)
    env.close()

    @classmethod
    def save(cls) :
        file = cls.root_dir + '/models/config.pkl'
        with open(file,'wb')  as fp :
            pickle.dump(cls,fp)
    
    @classmethod
    def load(cls) :
        file = cls.root_dir + '/models/config.pkl'
        with open(file,'rb') as f:
            config = pickle.load(f)
            cls.game = config.game
            cls.root_dir = config.root_dir
            cls.timesteps = config.timesteps
            cls.batch_size = config.batch_size
            cls.buffer_size = config.buffer_size
            cls.epochs = config.epochs
            cls.gradient_clip = config.gradient_clip
            cls.epsilon=config.epsilon
            cls.min_epsilon = config.min_epsilon
            cls.entropy = config.entropy
            cls.gamma = config.gamma
            cls.save_rate = config.save_rate
            cls.critic_learning_rate = config.critic_learning_rate
            cls.actor_learning_rate = config.actor_learning_rate
            cls.use_shuffle = config.use_shuffle
            cls.use_pixels = config.use_pixels
            if(cls.use_pixels) :
                cls.resized_height = config.resized_height
                cls.resized_width = config.resized_width
            cls.network_type = config.network_type
            #mlp
            cls.mlp_hidden_layers = config.mlp_hidden_layers
            cls.mlp_hidden_units = config.mlp_hidden_units
            #lstm
            if(cls.network_type == 'lstm') :
                cls.lstm_layers = config.lstm_layers
                cls.lstm_units = config.lstm_units
                cls.timestamps = config.timestamps
            #conv2d
            elif(cls.network_type == 'conv2d' and cls.use_pixels) :
                cls.conv_layers = config.conv_layers
                cls.conv_units = config.conv_units
                cls.stack_size = config.stack_size
            
            cls.env = config.env