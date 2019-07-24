from mario_environment import MarioEnvironment
from config import Config
from keras.models import model_from_json
import numpy as np
import sys
import keras.backend as K
import tensorflow as tf
import time

def simulate() :
    env = Config.env.clone()
    dummy_advantage = np.zeros((1, 1))
    dummy_old_actions_probs = np.zeros((1, env.getOutputSize()))
    episode = sys.argv[1]
    dir = Config.root_dir + '/models/episode-' +str(episode) + '/'
    with open(dir + 'actor.json','rb') as fp :
        json_model = fp.read()

    model = model_from_json(json_model)
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