
from config import Config
from termcolor import colored
import numpy as np
import os
import tensorflow as tf
from critic import Critic
from actor import Actor
import cv2
import matplotlib.pyplot as plt
from scipy import signal
#from pyvirtualdisplay import Display
#display = Display(visible=0, size=(1400, 900))
#display.start()

class PPO() :

    def __init__(self,sess) :

        self.sess = sess
        if(Config.start_episode > 0) :
            Config.load()
        self.env = Config.env.clone()
        self.critic = Critic(sess,self.env,'critic')
        self.actor = Actor(sess,self.env,'new_actor')
        init = tf.global_variables_initializer()
        sess.run(init)
        
        self.critic.init()
        self.actor.init()
        self.timesteps = 0
        
    

    def save(self,episode) :
        dir = Config.root_dir + '/models/episode-' + str(episode) + '/'
        if not os.path.exists(Config.root_dir + '/models') :
            os.mkdir(Config.root_dir + '/models')
            os.mkdir(dir)
        elif not os.path.exists(dir):
            os.mkdir(dir)
        self.actor.save(dir,episode)
        self.critic.save(dir,episode)
    
    def update_networks(self,batch) :
        states,rewards,mask,actions_probs,_ = batch
        
        estimated_rewards = self.critic.predict(states)
        rewards = rewards.reshape([rewards.shape[0],1])
        advantages = rewards - estimated_rewards
        advantages = (advantages - advantages.mean()) / np.maximum(advantages.std(), 1e-6)

        self.actor.train(states,advantages,actions_probs,mask)


        self.critic.train(states,rewards)

    def get_discounted_rewards(self,rewards,done):
        for j in range(len(rewards) - 2, -1, -1):
            rewards[j] += rewards[j + 1] * Config.gamma
        if(not done) :
            rewards = rewards[:-1]
        return rewards

    def collect_batch(self,state,next_state,total_rewards,episode) :
        states = []
        batch_advantages = []
        batch_states = []
        rewards = []
        batch_rewards = []
        batch_actions_probs = []
        mask = []
        for step in range(Config.buffer_size) :
            state = self.preprocess(state,next_state)
            states.append(state) 

            if(self.env.is_discrete) :
                actions_probs = self.actor.predict(np.expand_dims(state,axis=0))
                actions_probs = actions_probs.reshape([actions_probs.shape[1]])
                action = np.random.choice(range(len(actions_probs)),p=actions_probs)
                current_action = np.zeros(shape=actions_probs.shape)
                current_action[action] = 1
                
                mask.append(current_action)
                batch_actions_probs.append(actions_probs)
            else :
                action,action_probs = self.actor.predict(np.expand_dims(state,axis=0))
                action = action.reshape([action.shape[1]])
                action_probs = action_probs.reshape([action_probs.shape[1]])
                mask.append(action)
                batch_actions_probs.append(action_probs)

            
            next_state,reward,done,_ = self.env.step(action)
            rewards.append(self.reward_scaler(reward))
            total_rewards += reward
            

            if(done) :
                batch_rewards += self.get_discounted_rewards(rewards,True)
                batch_states += states
                states = []
                rewards = []
                if(episode != 0 and episode % Config.log_episodes == 0) :
                  average_rewards  = total_rewards / Config.log_episodes
                  data = 'episode {}/{} \t reward  {}'.format(episode,Config.episodes,average_rewards)
                  print(colored(data,'green'))
                  print(colored('timesteps : {}'.format(self.timesteps),'green'))
                  total_rewards = 0
                if(episode % Config.save_rate == 0) :
                    self.save(episode)
                episode += 1
                state = self.env.reset()
                next_state = None
            self.timesteps += 1

        if(len(states) > 0) :
            rewards = self.get_discounted_rewards(rewards,False)
            batch_rewards += rewards
            if(len(rewards) < len(states)) :
                states = states[:-1]
                mask = mask[:-1]
                batch_actions_probs = batch_actions_probs[:-1]
            batch_states += states

        batch = [np.array(batch_states),np.array(batch_rewards),np.array(mask),np.array(batch_actions_probs),np.array(batch_advantages)]
        if(episode >= Config.episodes) :
            end = True
        else :
            end = False
        return batch,end,episode,total_rewards,state,next_state

    def reward_scaler(self,reward) :
        if(Config.reward_scaler is None) :
            return reward
        if(Config.reward_scaler == 'positive') :
            return max(-0.001, reward / 100.0)
        if(Config.reward_scaler == 'scale') :
            return reward / 100

    def preprocess_pixels(self,state,next_state) :
        if(next_state is not None) :
            frame = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame,(Config.resized_width,Config.resized_height),interpolation=cv2.INTER_AREA) / 255
            if(Config.network_type == 'mlp' or Config.network_type == 'lstm') :
                frame = frame.reshape([Config.resized_width * Config.resized_height])
            elif(Config.network_type == 'conv2d') :
                frame = frame.reshape([Config.resized_width , Config.resized_height])
            else :
                raise Exception('Unable to preprocess state.Check config')
            if(Config.network_type == 'mlp') :
                return frame
            if(Config.network_type == 'lstm') :
                stack = state[1:,:]
                res = []
                for index in range(Config.timestamps) :
                    if(index == Config.timestamps - 1) :
                        res.append(frame)
                    else :
                        res.append(stack[index,:])
                return np.stack(res,axis=0)
            if(Config.network_type == 'conv2d') :
                stack = state[:,:,1:]
                res = []
                for index in range(Config.stack_size) :
                    if(index == Config.stack_size - 1) :
                        res.append(frame)
                    else :
                        res.append(stack[:,:,index])
                return np.stack(res,axis=2)
        else :
            frame = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame,(Config.resized_width,Config.resized_height),interpolation=cv2.INTER_AREA) / 255
            if(Config.network_type == 'mlp' or Config.network_type == 'lstm') :
                frame = frame.reshape([Config.resized_width * Config.resized_height])
            elif(Config.network_type == 'conv2d') :
                frame = frame.reshape([Config.resized_width , Config.resized_height])
            else :
                raise Exception('Unable to preprocess state.Check config')

            if(Config.network_type == 'mlp') :
                return frame
            if(Config.network_type == 'lstm') :
                return np.stack([frame for _ in range(Config.timestamps)],axis=0)
            if(Config.network_type == 'conv2d') :
                return np.stack([frame for _ in range(Config.stack_size)],axis=2)

    def preprocess(self,state,next_state) :
        if(Config.use_pixels) :
            return self.preprocess_pixels(state,next_state)
        elif(Config.network_type == 'lstm') :
            if(next_state is not None) :
                stack = state[1:,:]
                res = []
                for index in range(Config.timestamps) :
                    if(index == Config.timestamps - 1) :
                        res.append(next_state)
                    else :
                        res.append(stack[index,:])
                return np.stack(res,axis=0)
            else :
                return np.stack([state for _ in range(Config.timestamps)],axis=0)
        if(next_state is None) :
            return state
        return next_state

    def run(self) :
        Config.save()
        state = self.env.reset()
        next_state = None
        total_rewards = 0
        episode = Config.start_episode
        while(True) :
            batch,end,episode,total_rewards,state,next_state = self.collect_batch(state,next_state,total_rewards,episode)
            self.update_networks(batch)
            if(end) :
                break

            
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:   
    ppo = PPO(sess)
    ppo.run()