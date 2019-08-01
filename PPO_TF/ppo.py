
from config import Config
from termcolor import colored
import numpy as np
import os
import tensorflow as tf
from critic import Critic
from actor import Actor

#from pyvirtualdisplay import Display
#display = Display(visible=0, size=(1400, 900))
#display.start()



class PPO() :

    def __init__(self,sess) :

        self.sess = sess
        self.env = Config.env.clone()
        self.critic = Critic(sess,self.env.getInputSize(),self.env.getOutputSize(),self.env.use_pixels,'critic')
        self.actor = Actor(sess,self.env.getInputSize(),self.env.getOutputSize(),self.env.use_pixels,self.env.is_discrete,'new_actor')
        self.old_actor = Actor(sess,self.env.getInputSize(),self.env.getOutputSize(),self.env.use_pixels,self.env.is_discrete,'old_actor')
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        init = tf.global_variables_initializer()
        sess.run(init)
        
        self.old_actor.copyTrainables(self.actor.scope)
        
    
    def save(self,episode) :
        dir = Config.root_dir + '/models/episode-' + str(episode) + '/'
        if not os.path.exists(Config.root_dir + '/models') :
            os.mkdir(Config.root_dir + '/models')
            os.mkdir(dir)
        elif not os.path.exists(dir):
            os.mkdir(dir)
        with open(dir + "actor.json", "w") as json_file:
            actor_model_no_noise = self.buildActorNetwork(add_noise=False)
            json_file.write(actor_model_no_noise.to_json())
        self.actor_model.save_weights(dir + 'actor_weights')
    
    def updateNetworks(self,batch) :
        states,rewards,mask,actions_probs = batch
        
        estimated_rewards = self.critic.predict(states)
        rewards = rewards.reshape([rewards.shape[0],1])
        advantages = rewards - estimated_rewards

        '''actor_dataset = tf.data.Dataset.from_tensor_slices((states,advantages,actions_probs,mask))
        actor_dataset = actor_dataset.shuffle(len(states)).repeat().batch(Config.batch_size).map(lambda x,y,z,w: (x,y,z,w), num_parallel_calls=4).prefetch(buffer_size=int(len(states)/2))
        actor_dataset = actor_dataset.make_one_shot_iterator().get_next()
        critic_dataset = tf.data.Dataset.from_tensor_slices((states,rewards))
        critic_dataset = critic_dataset.shuffle(len(states)).repeat().batch(Config.batch_size).map(lambda x,y: (x,y), num_parallel_calls=4).prefetch(buffer_size=int(len(states)/2))
        critic_dataset = critic_dataset.make_one_shot_iterator().get_next()'''

        self.actor.train(states,advantages,actions_probs,mask)

        self.old_actor.copyTrainables(self.actor.scope)

        self.critic.train(states,rewards)


    def getDiscountedRewards(self,rewards,done):
        for j in range(len(rewards) - 2, -1, -1):
            rewards[j] += rewards[j + 1] * Config.gamma
        if(not done) :
            rewards = rewards[:-1]
        return rewards

    def collectBatch(self,state,next_state,total_rewards,episode) :
        states = []
        batch_states = []
        rewards = []
        batch_rewards = []
        batch_actions_probs = []
        mask = []
        for step in range(Config.buffer_size) :
            state = self.env.preprocess(state,next_state)
            states.append(state) 

            if(self.env.is_discrete) :
                actions_probs = self.old_actor.predict(np.expand_dims(state,axis=0))
                actions_probs = actions_probs.reshape([actions_probs.shape[1]])
                action = np.random.choice(range(len(actions_probs)),p=actions_probs)
                current_action = np.zeros(shape=actions_probs.shape)
                current_action[action] = 1
                
                mask.append(current_action)
                batch_actions_probs.append(actions_probs)
            else :
                action,action_probs = self.old_actor.predict(np.expand_dims(state,axis=0))
                action = action.reshape([action.shape[1]])
                action_probs = action_probs.reshape([action_probs.shape[1]])
                mask.append(action)
                batch_actions_probs.append(action_probs)

            
            next_state,reward,done,_ = self.env.step(action)
            rewards.append(reward)
            total_rewards += reward
            if(done) :
                batch_rewards += self.getDiscountedRewards(rewards,True)
                batch_states += states
                states = []
                rewards = []
                data = 'episode {}/{} \t reward  {}'.format(episode,Config.episodes,total_rewards)
                print(colored(data,'green'))
                total_rewards = 0
                #if(episode % Config.save_rate == 0) :
                #    self.save(episode)
                episode += 1
                state = self.env.reset()
                next_state = None

        if(len(states) > 0) :
            rewards = self.getDiscountedRewards(rewards,False)
            batch_rewards += rewards
            if(len(rewards) < len(states)) :
                states = states[:-1]
                mask = mask[:-1]
                batch_actions_probs = batch_actions_probs[:-1]
            batch_states += states

        batch = [np.array(batch_states),np.array(batch_rewards),np.array(mask),np.array(batch_actions_probs)]
        if(episode >= Config.episodes) :
            end = True
        else :
            end = False
        return batch,end,episode,total_rewards,state,next_state

    def run(self) :
        state = self.env.reset()
        next_state = None
        total_rewards = 0
        episode = 0
        while(True) :
            batch,end,episode,total_rewards,state,next_state = self.collectBatch(state,next_state,total_rewards,episode)
            self.updateNetworks(batch)
            if(end) :
                break
    
            
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:   
    ppo = PPO(sess)
    ppo.run()
