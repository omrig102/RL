from mario_environment import MarioEnvironment
from gym_environment import GymEnvironment
from keras.models import model_from_json
from noisy_gaussian import NoisyDense
import numpy as np
import sys
import keras.backend as K
import tensorflow as tf
import time

def simulate() :
    env = GymEnvironment('LunarLander-v2',13,13,use_pixels=False,stack_size=4,is_discrete=True,save_video=False,save_video_interval=10)
    #env = MarioEnvironment('SuperMarioBros-v0',48,48,use_pixels=True,stack_size=4,is_discrete=True,save_video=False,save_video_interval=5)
    dummy_advantage = np.zeros((1, 1))
    dummy_old_actions_probs = np.zeros((1, env.getOutputSize()))
    episode = sys.argv[1]
    dir = '.' + '/models/episode-' +str(episode) + '/'
    with open(dir + 'actor.json','rb') as fp :
        json_model = fp.read()

    model = model_from_json(json_model,custom_objects={'NoisyDense': NoisyDense})
    model.load_weights(dir + 'actor_weights')
    model.summary()
    state = env.reset()
    next_state = None
    total_rewards = 0
    while(True) :
        env.render()
        #time.sleep(0.02)
        state = env.preprocess(state,next_state)
        action_probs = model.predict(np.expand_dims(state,0))
        action_probs = action_probs.reshape([action_probs.shape[1]])
        if(env.is_discrete) :
            action = np.random.choice(range(len(action_probs)),p=action_probs)
            #action = np.argmax(action_probs)
        else :
            action = action_probs[0]
        next_state,reward,done,_ = env.step(action)
        total_rewards += reward
        #state = next_state
        if(done) :
            print('Game Ended! Total Rewards {}'.format(total_rewards))
            break


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

simulate()
