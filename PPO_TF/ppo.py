
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
        if(Config.start_timestep > 0) :
            Config.load()
        self.env = Config.env.clone()
        if(Config.policy_type == 'actor_critic') :
            self.actor_critic = ActorCritic(sess,self.env,'actor_critic')
        else :
            self.critic = Critic(sess,self.env,'critic')
            self.actor = Actor(sess,self.env,'new_actor')
        init = tf.global_variables_initializer()
        sess.run(init)
        
        if(Config.with_summary) :
            self.writer = tf.summary.FileWriter(Config.root_dir + Config.log_dir, sess.graph)
        else :
            self.writer = None

        if(Config.policy_type == 'actor_critic') :
            self.actor_critic.init(self.writer)
        else :
            self.critic.init()
            self.actor.init()
        
    

    def save(self,current_timestep) :
        dir = Config.root_dir + '/models/timestep-' + str(current_timestep) + '/'
        if not os.path.exists(Config.root_dir + '/models') :
            os.mkdir(Config.root_dir + '/models')
            os.mkdir(dir)
        elif not os.path.exists(dir):
            os.mkdir(dir)
        if(Config.policy_type == 'actor_critic') :
            self.actor_critic.save(dir,current_timestep)
        else :
            self.actor.save(dir,current_timestep)
            self.critic.save(dir,current_timestep)

    def update_networks(self,batch,current_timestep) :
        current_update = current_timestep / Config.buffer_size
        total_updates = Config.timesteps / Config.buffer_size
        frac = 1.0 - (current_update - 1.0) / total_updates
        lr = Config.actor_learning_rate(frac)
        states,rewards,actions,actions_probs,advantages,v_states = batch
        
        if(Config.policy_type == 'actor_critic') :
             estimated_rewards = v_states.reshape(v_states.shape[0],1)
        else :
            estimated_rewards = self.critic.predict(states)
        advantages = advantages.reshape([advantages.shape[0],1])
        rewards = rewards.reshape([rewards.shape[0],1])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        

        if(Config.policy_type == 'actor_critic') :
            self.actor_critic.train(states,advantages,actions_probs,actions,rewards,estimated_rewards,lr)
        else :
            self.actor.train(states,advantages,actions_probs,actions)
            self.critic.train(states,rewards,estimated_rewards)
        

    def get_discounted_rewards_gae(self,states,rewards,dones) :
        if(Config.policy_type == 'actor_critic') :
            v_states = self.actor_critic.predict_value(states)
        else :
            v_states = self.critic.predict(states)
        advantages = np.zeros_like(rewards)
        v_states = v_states.reshape([v_states.shape[0]])
        lastgaelam = 0
        for t in reversed(range(Config.buffer_size)):
            if t == Config.buffer_size - 1:
                nextnonterminal = 1.0 - dones[-1]
                nextvalues = v_states[-1]
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = v_states[t+1]
            delta = rewards[t] + Config.gamma * nextvalues * nextnonterminal - v_states[t]
            advantages[t] = lastgaelam = delta + Config.gamma * Config.gae * nextnonterminal * lastgaelam
        
        values = np.asarray(rewards, dtype=np.float32)
        discounted_rewards = advantages + v_states

        return discounted_rewards,advantages,v_states

    def collect_batch(self,state,next_state,total_rewards,current_timestep,episode) :
        batch_advantages = []
        batch_states = []
        batch_rewards = []
        batch_actions_probs = []
        actions = []
        dones = []
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
                
                actions.append(current_action)
                batch_actions_probs.append(actions_probs)
            else :
                action,action_probs = self.actor.predict(np.expand_dims(state,axis=0))
                action = action.reshape([action.shape[1]])
                action_probs = action_probs.reshape([action_probs.shape[1]])
                actions.append(action)
                batch_actions_probs.append(action_probs)

            
            next_state,reward,done,_ = self.env.step(action)
            dones.append(done)
            batch_rewards.append(self.reward_scaler(reward))
            total_rewards += reward
            

            if(done) :
                if(episode != 0 and episode % (Config.log_episodes)  == 0) :
                    average_rewards  = total_rewards / Config.log_episodes
                    data = 'timestep {}/{} \t reward  {}'.format(current_timestep,Config.timesteps,average_rewards)
                    print(colored(data,'green'))
                    total_rewards = 0
                if(episode % Config.save_rate == 0) :
                    self.save(current_timestep)
                
                episode += 1
                state = self.env.reset()
                next_state = None
            current_timestep += 1

        batch_rewards,batch_advantages,v_states = self.get_discounted_rewards_gae(batch_states,batch_rewards,dones)

        batch = [np.array(batch_states),np.array(batch_rewards),np.array(actions),np.array(batch_actions_probs),np.array(batch_advantages),np.array(v_states)]
        
        if(current_timestep >= Config.timesteps) :
            end = True
        else :
            end = False
        return batch,end,current_timestep,total_rewards,state,next_state,episode

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
        current_timestep = Config.start_timestep
        episode = 0
        while(True) :
            batch,end,current_timestep,total_rewards,state,next_state,episode = self.collect_batch(state,next_state,total_rewards,current_timestep,episode)
            self.update_networks(batch,current_timestep)
            if(end) :
                break

            
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:   
    ppo = PPO(sess)
    ppo.run()