from keras.layers import Dense,Input,Add,GaussianNoise,Activation,Conv2D,Flatten,BatchNormalization,Reshape
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from config import Config
from termcolor import colored
import numpy as np
from noisy_gaussian import NoisyDense
import os
import tensorflow as tf

#from pyvirtualdisplay import Display
#display = Display(visible=0, size=(1400, 900))
#display.start()

def ppoLoss(advantage,old_actions_probs) :
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_actions_probs
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - Config.epsilon, max_value=1 + Config.epsilon) * advantage))
    return loss

class PPO() :

    def __init__(self) :
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        K.set_session(sess)
        K.set_learning_phase(1)
        self.env = Config.env.clone()
        self.critic_model = self.buildCriticNetwork()
        self.actor_model = self.buildActorNetwork()
        self.old_actor_model = self.buildActorNetwork()
        self.old_actor_model.set_weights(self.actor_model.get_weights())
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_actions_probs = np.zeros((1, self.env.getOutputSize()))
        
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
            

    def buildCriticNetwork(self) :
        critic_input = Input(shape=self.env.getInputSize())
        current_layer = critic_input
        if(self.env.use_pixels and Config.use_conv_layers) :
            #normalize_1 = BatchNormalization()(critic_input)
            current_layer = Conv2D(filters=48,kernel_size=[3,3],activation='relu')(current_layer)
            current_layer = Conv2D(filters=48,kernel_size=[3,3],activation='relu')(current_layer)
            current_layer = Flatten()(current_layer)
            #current_layer = BatchNormalization()(current_layer)
        elif(self.env.use_pixels) :
            current_layer = Reshape(target_shape=[self.env.getInputSize()[0] * self.env.getInputSize()[1]])(current_layer)
        for _ in range(Config.hidden_size) :
            current_layer = Dense(units=Config.hidden_units,activation='relu')(current_layer)
        critic_output = Dense(units=1,activation='linear')(current_layer)

        model = Model(critic_input,critic_output)
        model.compile(optimizer=Adam(lr=Config.critic_learning_rate),loss='mse')


        return model

    def buildActorNetwork(self,add_noise=True) :
        advantage = Input(shape=[1])
        old_actions_probs = Input(shape=[self.env.getOutputSize()])
        state = Input(shape=self.env.getInputSize())
        current_layer = state
        if(self.env.use_pixels and Config.use_conv_layers) :
            #normalize_1 = BatchNormalization()(current_layer)
            current_layer = Conv2D(filters=48,kernel_size=[3,3],activation='relu')(current_layer)
            current_layer = Conv2D(filters=48,kernel_size=[3,3],activation='relu')(current_layer)
            current_layer = Flatten()(current_layer)
            #current_layer = BatchNormalization()(current_layer)
        elif(self.env.use_pixels) :
            current_layer = Reshape(target_shape=[self.env.getInputSize()[0] * self.env.getInputSize()[1]])(current_layer)
        
        for _ in range(Config.hidden_size) :
            current_layer = Dense(units=Config.hidden_units,activation='relu')(current_layer)
        
        actor_outputs = NoisyDense(units=self.env.getOutputSize(),training=add_noise)(current_layer)

        #if(add_noise) :
        #    actor_outputs = GaussianNoise(stddev=0.1)(actor_outputs)
        if(self.env.is_discrete) :
            activation = 'softmax'
            loss = ppoLoss(advantage=advantage,old_actions_probs=old_actions_probs)
        else :
            activation = 'tanh'
            loss = ppoLoss(advantage=advantage,old_actions_probs=old_actions_probs)
        actor_outputs = Activation(activation)(actor_outputs)
        

        model = Model([state,old_actions_probs,advantage],actor_outputs)

        model.compile(optimizer=Adam(lr=Config.actor_learning_rate),loss=loss)
        
        return model

    def updateNetworks(self,batch) :
        states,rewards,mask,actions_probs = batch
        
        estimated_rewards = self.critic_model.predict_on_batch(states)
        rewards = rewards.reshape([rewards.shape[0],1])
        advantages = rewards - estimated_rewards
        
        self.actor_model.fit([states,actions_probs,advantages],[mask], verbose=False,batch_size=Config.batch_size,shuffle=True,epochs=Config.epochs)
        
        self.old_actor_model.set_weights(self.actor_model.get_weights())
        self.critic_model.fit([states],[rewards], verbose=False,batch_size=Config.batch_size,shuffle=True,epochs=Config.epochs)

    def getDiscountedRewards(self,rewards,done):
        for j in range(len(rewards) - 2, -1, -1):
            rewards[j] += rewards[j + 1] * Config.gamma
        if(not done) :
            rewards = rewards[:-1]
        return rewards

    def updateRewards(self,rewards,done) :
        return self.getDiscountedRewards(rewards,done)
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
            actions_probs = self.old_actor_model.predict([np.expand_dims(state,axis=0),self.dummy_old_actions_probs,self.dummy_advantage])
            actions_probs = actions_probs.reshape([actions_probs.shape[1]])

            batch_actions_probs.append(actions_probs)
            
            if(self.env.is_discrete) :
                action = np.random.choice(range(len(actions_probs)),p=actions_probs)
                current_action = np.zeros(shape=[self.env.getOutputSize()])
                current_action[action] = 1
                mask.append(current_action)
            else :
                action = actions_probs
                mask.append(action)
            
            next_state,reward,done,_ = self.env.step(action)
            rewards.append(reward)
            total_rewards += reward
            if(done) :
                batch_rewards += self.updateRewards(rewards,True)
                batch_states += states
                states = []
                rewards = []
                data = 'episode {}/{} \t reward  {}'.format(episode,Config.episodes,total_rewards)
                print(colored(data,'green'))
                total_rewards = 0
                if(episode % Config.save_rate == 0) :
                    self.save(episode)
                episode += 1
                state = self.env.reset()
                next_state = None

        if(len(states) > 0) :
            rewards = self.updateRewards(rewards,False)
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
    
            
            
            
ppo = PPO()
ppo.run()
