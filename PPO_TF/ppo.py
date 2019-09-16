
from config import Config
from termcolor import colored
import numpy as np
import os
import tensorflow as tf
from critic import Critic
from actor import Actor
from actor_critic import ActorCritic
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
        if(Config.policy_type == 'actor_critic') :
            self.actor_critic = ActorCritic(sess,self.env,'actor_critic')
        else :
            self.critic = Critic(sess,self.env,'critic')
            self.actor = Actor(sess,self.env,'new_actor')
        init = tf.global_variables_initializer()
        sess.run(init)
        
        if(Config.policy_type == 'actor_critic') :
            self.actor_critic.init()
        else :
            self.critic.init()
            self.actor.init()
        self.timesteps = 0
        self.epsilon_decay_step = (Config.epsilon - Config.min_epsilon) / Config.episodes
        
    

    def save(self,episode) :
        dir = Config.root_dir + '/models/episode-' + str(episode) + '/'
        if not os.path.exists(Config.root_dir + '/models') :
            os.mkdir(Config.root_dir + '/models')
            os.mkdir(dir)
        elif not os.path.exists(dir):
            os.mkdir(dir)
        if(Config.policy_type == 'actor_critic') :
            self.actor_critic.save(dir,episode)
        else :
            self.actor.save(dir,episode)
            self.critic.save(dir,episode)

    def update_networks(self,batch) :
        states,rewards,mask,actions_probs,advantages = batch
        
        if(Config.policy_type == 'actor_critic') :
             estimated_rewards = self.actor_critic.predict_value(states)
        else :
            estimated_rewards = self.critic.predict(states)
        #rewards = rewards.reshape([rewards.shape[0],1])
        #advantages = rewards - estimated_rewards
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        advantages = advantages.reshape([advantages.shape[0],1])
        rewards = rewards.reshape([rewards.shape[0],1])

        if(Config.policy_type == 'actor_critic') :
            self.actor_critic.train(states,advantages,actions_probs,mask,rewards,estimated_rewards)
        else :
            self.actor.train(states,advantages,actions_probs,mask)
            self.critic.train(states,rewards,estimated_rewards)
        
        Config.epsilon -= self.epsilon_decay_step

    def discount_cumsum(self,x, discount):
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def get_discounted_rewards_gae(self,states,rewards,dones) :
        if(Config.policy_type == 'actor_critic') :
            v_states = self.actor_critic.predict_value(states)
        else :
            v_states = self.critic.predict(states)
        advantages = np.zeros_like(rewards)
        v_states = v_states.reshape([v_states.shape[0]])
        rewards = np.array(rewards)
        lastgaelam = 0
        for t in reversed(range(Config.buffer_size)):
            if(t == Config.buffer_size - 1) :
                nextnonterminal = 1 - dones[-1]
                nextvalues = v_states[-1]
            else :
                nextnonterminal = 1 - dones[t+1]
                nextvalues = v_states[t+1]
            delta = rewards[t] + Config.gamma * nextvalues * nextnonterminal - v_states[t]
            advantages[t] = lastgaelam = delta + Config.gamma * Config.gae * nextnonterminal * lastgaelam
        
        
        discounted_rewards = advantages + v_states

        return discounted_rewards,advantages

    def get_discounted_rewards(self,rewards,done):
        for j in range(len(rewards) - 2, -1, -1):
            rewards[j] += rewards[j + 1] * Config.gamma
        if(not done) :
            rewards = rewards[:-1]
        return rewards

    def collect_batch(self,state,next_state,total_rewards,episode) :
        batch_advantages = []
        batch_states = []
        batch_rewards = []
        batch_dones = []
        batch_actions_probs = []
        mask = []
        for step in range(Config.buffer_size) :
            state = self.preprocess(state,next_state)
            batch_states.append(state) 

            if(self.env.is_discrete) :
                if(Config.policy_type == 'actor_critic') :
                    actions_probs = self.actor_critic.predict_action(np.expand_dims(state,axis=0))
                else :
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
            batch_dones.append(done)
            batch_rewards.append(self.reward_scaler(reward))
            total_rewards += reward
            

            if(done) :
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


        batch_rewards,batch_advantages = self.get_discounted_rewards_gae(batch_states,batch_rewards,batch_dones)

        batch = [np.array(batch_states),np.array(batch_rewards),np.array(mask),np.array(batch_actions_probs),np.array(batch_advantages)]
        if(episode >= Config.episodes) :
            end = True
        else :
            end = False
        return batch,end,episode,total_rewards,state,next_state

    def reward_scaler(self,reward) :
        if(Config.reward_scaler is None) :
            return reward
        if(Config.reward_scaler == 'sign') :
            return np.sign(reward)
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