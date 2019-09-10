from gym_environment import GymEnvironment
from mario_environment import MarioEnvironment
from retro_environment import RetroEnvironment
import pickle


class Config() :

    #game = 'SuperMarioBros-v0'
    game = 'BipedalWalkerHardcore-v2'
    root_dir = '.'
    episodes = 1000000
    start_episode = 0
    save_video = True
    save_video_interval = 50
    save_rate = 1000
    log_episodes = 100
    batch_size = 1024
    buffer_size = 4096
    epochs = 10
    epsilon=0.05
    entropy = 0.1
    gamma = 0.99
    reward_scaler = 'positive'
    
    critic_learning_rate = 0.0001
    actor_learning_rate = 0.0001
    use_shuffle = True
    use_pixels = False
    if(use_pixels) :
        resized_height = 84
        resized_width = 84
    network_type = 'mlp'
    #mlp
    mlp_hidden_layers = 3
    mlp_hidden_units = [128,128,128]
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
            cls.episodes = config.episodes
            cls.batch_size = config.batch_size
            cls.buffer_size = config.buffer_size
            cls.epochs = config.epochs
            cls.epsilon=config.epsilon
            cls.entropy = config.entropy
            cls.gamma = config.gamma
            cls.l2 = config.l2
            cls.save_rate = config.save_rate
            cls.critic_learning_rate = config.critic_learning_rate
            cls.actor_learning_rate = config.actor_learning_rate
            cls.sigma_limit = config.sigma_limit
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
