import tensorflow as tf
from config import Config
import numpy as np
import math
import models
import os

class Actor() :

    def __init__(self,sess,env,scope) :
        self.scope = scope
        self.env = env
        self.input_size = [None]
        if(Config.network_type == 'lstm') :
            self.input_size.append(Config.timestamps)
        if(Config.use_pixels) :
            if(Config.network_type == 'mlp' or Config.network_type == 'lstm') :
                self.input_size.append(Config.resized_height * Config.resized_width)
            else :
                self.input_size.append(Config.resized_height)
                self.input_size.append(Config.resized_width)
                self.input_size.append(Config.stack_size)
        else :
            for shape in self.env.get_input_size() :
                self.input_size.append(shape)
        
        self.sess = sess
        with tf.variable_scope(scope) as s:
                self.build_actor_network()

    def init(self) :
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        if(Config.start_episode > 0) :
            self.load()

    def load(self) :
        dir = Config.root_dir + '/models/episode-' + str(Config.start_episode) + '/actor-' + self.scope + '/model.ckpt-' + str(Config.start_episode)
        
        self.saver.restore(self.sess,dir)

    def save(self,dir,episode) :
        self.saver.save(self.sess,dir + '/actor-' + self.scope + '/model.ckpt',global_step=episode)

    def build_actor_network(self) :
        if(self.env.is_discrete) :
            self.build_actor_network_discrete()
        else :
            self.build_actor_network_continuous()

    def build_actor_network_continuous(self) :
        self.advantage = tf.placeholder(shape=[None,1],dtype=tf.float32,name='advantage')
        self.old_probs = tf.placeholder(shape=[None,self.env.get_output_size()],dtype=tf.float32,name='old_prob')
        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32,name='state')
        self.chosen_action = tf.placeholder(shape=[None,self.env.get_output_size()],dtype=tf.float32,name='chosen_action')
        
        

        mu = self.build_base_network(self.state,self.env.get_output_size(),tf.nn.tanh)
        
        high = Config.env.env.action_space.high
        low = Config.env.env.action_space.low
        
        log_sigma = tf.Variable(-0.5 * np.ones(self.env.get_output_size(), dtype=np.float32))
        sigma = tf.exp(log_sigma)

        dist = tf.contrib.distributions.Normal(mu,sigma)
        self.action = tf.clip_by_value(dist.sample(),low,high,name='action')
        self.probs = dist.prob(self.chosen_action,name='probs')

        self.loss_continuous()

    def build_base_network(self,x,output_size,output_activation,output_name=None) :
        if(Config.use_pixels) :
            if(Config.network_type == 'mlp') :
                return models.create_network_pixels_mlp(x,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name)
            elif(Config.network_type == 'conv2d') :
                return models.create_network_pixels_conv(x,Config.conv_layers,Config.conv_units,tf.nn.relu,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name)
            elif(Config.network_type == 'lstm')  :
                return models.create_network_lstm(x,Config.lstm_layers,Config.lstm_units,Config.unit_type,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name)
        elif(Config.network_type == 'mlp') :
            return models.create_mlp_network(x,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name)
        elif(Config.network_type == 'lstm') :
            return models.create_network_lstm(x,Config.lstm_layers,Config.lstm_units,Config.unit_type,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name)
        else :
            raise Exception('Unable to create base network,check config')



    def loss_continuous(self) :
        ratio = self.probs / tf.maximum(1e-10,self.old_probs)
        unclipped = ratio * self.advantage
        clipped = tf.clip_by_value(ratio,1-Config.epsilon,1+Config.epsilon) * self.advantage
        loss = -tf.reduce_mean(tf.minimum(unclipped,clipped))
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.actor_learning_rate)
        self.optimizer = optimizer.minimize(loss)

    def loss_discrete(self) :
        prob = tf.log(self.mask * self.outputs + 1e-10)
        old_prob = tf.log(self.mask * self.old_probs + 1e-10)
        ratio = tf.exp(prob - old_prob)
        unclipped = ratio * self.advantage
        clipped = tf.clip_by_value(ratio,1-Config.epsilon,1+Config.epsilon) * self.advantage
        loss = -tf.reduce_mean(tf.minimum(unclipped,clipped) + Config.entropy * -(self.mask * self.outputs * prob))
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.actor_learning_rate)
        self.optimizer = optimizer.minimize(loss)

    def build_actor_network_discrete(self) :
        self.mask = tf.placeholder(shape=[None,self.env.get_output_size()],dtype=tf.float32,name='mask')
        self.advantage = tf.placeholder(shape=[None,1],dtype=tf.float32,name='advantage')
        self.old_probs = tf.placeholder(shape=[None,self.env.get_output_size()],dtype=tf.float32,name='old_probs')
        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32,name='state')
        
        self.outputs = self.build_base_network(self.state,self.env.get_output_size(),tf.nn.softmax,'outputs',Config.l2,use_noise=False)
        

        self.loss_discrete()


    def predict(self,state) :
        if(self.env.is_discrete) :
            return self.sess.run(self.outputs,feed_dict={self.state:state})
        else :
            action = self.sess.run(self.action,feed_dict={self.state:state})
            #action = action[0]
            action_probs = self.sess.run(self.probs,feed_dict={self.state:state,self.chosen_action:action})
            return action,action_probs

    def train(self,states,advantages,old_probs,masks) :

        randomize = np.arange(len(states))
        for _ in range(Config.epochs) :
            if(Config.use_shuffle and Config.network_type != 'lstm') :
                np.random.shuffle(randomize)
            for index in range(int(Config.buffer_size/Config.batch_size)) :
                
                batch_states,batch_advantages,batch_old_probs,batch_masks = self.prepare_batch(states,advantages,old_probs,masks,index,randomize)
                if(self.env.is_discrete) :
                    self.__train_discrete(batch_states,batch_advantages,batch_old_probs,batch_masks)
                else :
                    self.__train_continuous(batch_states,batch_advantages,batch_old_probs,batch_masks)

    def __train_discrete(self,batch_states,batch_advantages,batch_old_probs,batch_masks) :
        self.sess.run(self.optimizer,feed_dict={self.state:batch_states,self.advantage:batch_advantages
                    ,self.old_probs:batch_old_probs,self.mask:batch_masks})

    def __train_continuous(self,batch_states,batch_advantages,batch_old_probs,batch_actions) :
        self.sess.run(self.optimizer,feed_dict={self.state:batch_states,self.advantage:batch_advantages
                    ,self.old_probs:batch_old_probs,self.chosen_action:batch_actions})

    def copy_trainables(self,actor_scope) :
        e1_params = [t for t in tf.trainable_variables(actor_scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables(self.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)
        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        self.sess.run(update_ops)


    def prepare_batch(self,states,advantages,old_probs,masks,current_batch,randomize) :
        random_states = states[randomize].copy()
        random_advantages = advantages[randomize].copy()
        random_old_probs = old_probs[randomize].copy()
        random_masks = masks[randomize].copy()

        current_index = int(current_batch * Config.batch_size)
        batch_states = random_states[current_index : current_index + Config.batch_size]
        batch_advantages = random_advantages[current_index : current_index + Config.batch_size]
        batch_old_probs = random_old_probs[current_index : current_index + Config.batch_size]
        batch_masks = random_masks[current_index : current_index + Config.batch_size]

        return batch_states,batch_advantages,batch_old_probs,batch_masks