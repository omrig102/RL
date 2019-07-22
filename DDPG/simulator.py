from keras.models import load_model
from config import Config
from gym_environment import GymEnvironment
import tensorflow as tf
import keras.backend as K
import numpy as np
import sys

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

episode = sys.argv[1]
dir = Config.root_dir + '/models/episode-' + str(episode) + '/'
model = load_model(dir + 'actor.h5')

env = GymEnvironment('BipedalWalker-v2',13,13,use_pixels=False)
env.initialize()
frame = env.reset()
frame = env.preprocess(frame)
frame = np.reshape(frame,newshape=[1,frame.shape[0]])
done = False
total_rewards = 0
while(not done) :
    env.render()
    action = env.actionProcessor(model.predict(frame))
    print('Action : {}'.format(action))
    new_frame,reward,done,info = env.step(action)
    total_rewards += reward
    new_frame = env.preprocess(new_frame)
    new_frame = np.reshape(new_frame,newshape=[1,new_frame.shape[0]])
    frame = new_frame
print('Game Ended : {}'.format(total_rewards))