from config import Config
from noise import OrnsteinUhlenbeckActionNoise
from _collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Add,BatchNormalization,Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf
import keras.backend as K
import numpy as np
import random
import os
from termcolor import colored

class DDPG() :

    def __init__(self) :
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        K.set_session(sess)
        self.sess = sess
        self.epsilon = Config.epsilon
        self.memory = deque(maxlen=Config.memory_size)
        self.env = Config.env.clone()

    def init(self) :
        self.actor_state_input,self.actor_model = self.buildActorModel()
        _,self.target_actor_model = self.buildActorModel()
        self.critic_state_input,self.critic_action_input,self.critic_model = self.buildCriticModel()
        _,_,self.target_critic_model = self.buildCriticModel()

        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())
        self.actor_critic_grad = tf.placeholder(tf.float32,shape=[None,self.env.getOutputSize()])

        actor_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,actor_weights,-self.actor_critic_grad)
        #self.actor_grads,_ = tf.clip_by_global_norm(self.actor_grads,clip_norm=1)
        grads = zip(self.actor_grads,actor_weights)
        
        self.optimize = tf.train.AdamOptimizer(learning_rate=Config.learning_rate_actor).apply_gradients(grads)

        self.critic_grads = tf.gradients(self.critic_model.output,self.critic_action_input)
        self.sess.run(tf.global_variables_initializer())

    def buildActorModel(self) :
        input_size = self.env.getInputSize()
        output_size = self.env.getOutputSize()

        state_input = Input(shape=[input_size])
        #actor_normalize_1 = BatchNormalization(trainable=True)(state_input)
        dense_1 = Dense(units=300,init='he_uniform',activation='selu')(state_input)
        dense_2 = Dense(units=300,init='he_uniform',activation='selu')(dense_1)
        output = Dense(units=output_size,activation='linear')(dense_2)

        model = Model(state_input,output)
        #model.compile(optimizer='adam',loss='mse')

        model.summary()
        return state_input,model

    def buildCriticModel(self) :
        state_input_size = self.env.getInputSize()
        action_input_size = self.env.getOutputSize()
        state_input = Input(shape=[state_input_size],name='state_input')
        #critic_normalize_1 = BatchNormalization(trainable=True)(state_input)
        state_dense = Dense(units=200,init='he_uniform',activation='selu')(state_input)
        action_input = Input(shape=[action_input_size],name='action_input')
        #critic_normalize_2 = BatchNormalization(trainable=True)(action_input)
        action_dense = Dense(units=200,init='he_uniform',activation='selu')(action_input)

        combine = Add()([state_dense,action_dense])
        critic_dense_1 = Dense(units=200,init='he_uniform',activation='selu')(combine)
        output = Dense(units=1,init='he_uniform',activation='linear')(critic_dense_1)

        model = Model([state_input,action_input],output)
        model.compile(optimizer=Adam(lr=Config.learning_rate_critic,decay=0.001),loss='mse')

        model.summary()
        return state_input,action_input,model

    def trainActor(self,train_data) :
        states = []
        for state,action,reward,done,new_state in train_data :
            states.append(state.reshape([state.shape[1]]))
            
        predicted_actions = self.actor_model.predict_on_batch(np.asarray(states))
        new_predicted_actions = []
        for action in predicted_actions :
            new_predicted_actions.append(np.clip(action,self.env.env.action_space.low,self.env.env.action_space.high))
        predicted_actions = new_predicted_actions
        grads = self.sess.run(self.critic_grads,feed_dict={self.critic_state_input : states,self.critic_action_input : predicted_actions})[0]
        self.sess.run(self.optimize,feed_dict={self.actor_state_input:states,self.actor_critic_grad:grads})
        

    def trainTargetActor(self) :
        actor_weights = np.array(self.actor_model.get_weights())
        actor_target_weights = np.array(self.target_actor_model.get_weights())
        actor_target_weights = Config.TAU * actor_weights + (1 - Config.TAU)* actor_target_weights
        
        self.target_actor_model.set_weights(actor_target_weights)

    def trainCritic(self,train_data) :
        states = []
        new_states = []
        actions = []
        targets = []
        for state,action,reward,done,new_state in train_data :
            
            states.append(state.reshape([state.shape[1]]))
            new_states.append(new_state.reshape([new_state.shape[1]]))
            actions.append(action.reshape([action.shape[1]]))
        
        next_actions = self.target_actor_model.predict_on_batch(np.array(new_states))
        new_next_actions = []
        for action in next_actions :
            new_next_actions.append(np.clip(action,self.env.env.action_space.low,self.env.env.action_space.high))
        next_actions = new_next_actions
        #next_actions = np.clip(next_actions,self.env.env.action_space.low,self.env.env.action_space.high)
        next_Qs = self.target_critic_model.predict_on_batch([np.array(new_states),np.array(next_actions)])
        #next_Qs = next_Qs.reshape([next_Qs.shape[0]])
        index = 0
        for state,action,reward,done,new_state in train_data :
            if(not done) :
                targets.append(reward + Config.gamma * next_Qs[index])
            else :
                targets.append(reward)
            index += 1
        self.critic_model.train_on_batch([np.array(states),np.array(actions)],np.array(targets))
        

    def trainTargetCritic(self) :
        critic_weights = np.array(self.critic_model.get_weights())
        critic_target_weights = np.array(self.target_critic_model.get_weights())
        critic_target_weights = Config.TAU * critic_weights + (1 - Config.TAU)* critic_target_weights
        
        self.target_critic_model.set_weights(critic_target_weights)
    
    
    def update(self) :
        if(len(self.memory) < Config.batch_size) :
            return
        
        train_data = random.sample(self.memory,Config.batch_size)
        self.trainCritic(train_data)
        self.trainActor(train_data)

    def updateTarget(self) :
        self.trainTargetActor()
        self.trainTargetCritic()
        
    
      

    def run(self) :
        self.init()
        self.env.initialize()

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.env.output_size),sigma=Config.sigma, theta=Config.sigma, dt=Config.dt)
        print('Initializing memory to : {}'.format(Config.initial_memory_size))
        frame = self.env.reset()
        frame = self.env.preprocess(frame)
        frame = np.reshape(frame,newshape=[1,frame.shape[0]])
        while(len(self.memory) < Config.initial_memory_size) :
          #if(len(self.memory) % 100 == 0) :
          #  print('Memory : {}/{}'.format(len(self.memory),Config.initial_memory_size))
          action = self.env.env.action_space.sample()
          new_frame,reward,done,_ = self.env.step(action)
          
          new_frame = self.env.preprocess(new_frame)
          new_frame = np.reshape(new_frame,newshape=[1,new_frame.shape[0]])
          action = action.reshape([1,action.shape[0]])
          self.memory.append((frame,action,reward,done,new_frame))
          
          if(done) :
            frame = self.env.reset()
            frame = self.env.preprocess(frame)
            frame = np.reshape(frame,newshape=[1,frame.shape[0]])
          else :
            frame = new_frame
          
        print('Done memory initialization')
        for episode in range(Config.episodes) :
            #actor_noise.reset()
            frame = self.env.reset()
            frame = self.env.preprocess(frame)
            frame = np.reshape(frame,newshape=[1,frame.shape[0]])
            episode_total_reward = 0
            for step in range(Config.steps) :
                
                if(Config.render) :
                    self.env.render()
                
                action = self.env.actionProcessor(self.actor_model.predict(frame))
                #action += actor_noise()
                action += np.clip(actor_noise(),self.env.env.action_space.low,self.env.env.action_space.high)
                action = np.clip(action,self.env.env.action_space.low,self.env.env.action_space.high)
                new_frame,reward,done,_ = self.env.step(action)
                new_frame = self.env.preprocess(new_frame)
                new_frame = np.reshape(new_frame,newshape=[1,new_frame.shape[0]])
                episode_total_reward += reward
                action = action.reshape([1,action.shape[0]])
                self.memory.append((frame,action,reward,done,new_frame))
                frame = new_frame
                
                if(done) :
                    data = 'episode {}/{} \t reward  {}'.format(episode,Config.episodes,episode_total_reward)
                    print(colored(data,'green'))
                    break
                if(step % Config.update_rate == 0) :
                    self.update()
                    self.updateTarget()
                    
            
            if(episode % Config.save_rate == 0) :
                dir = Config.root_dir + '/models/episode-' + str(episode) + '/'
                if not os.path.exists(dir):
                    os.mkdir(dir)
                self.target_actor_model.save(dir + 'actor.h5')
                self.target_critic_model.save(dir + 'critic.h5')

            #self.epsilon -= 1.0 / (Config.episodes/2)

