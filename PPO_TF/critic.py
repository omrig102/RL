import tensorflow as tf
from config import Config
import numpy as np
import models
import os

class Critic() :

    def __init__(self,sess,env,scope) :
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
        self.scope = scope
        
        with tf.variable_scope(scope) as s:
            self.build_critic_network()
    
        

    def init(self) :
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        if(Config.start_episode > 0) :
            self.load()

    def load(self) :
        dir = Config.root_dir + '/models/episode-' + str(Config.start_episode) + '/critic/model.ckpt-' + str(Config.start_episode)
        #ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        self.saver.restore(self.sess,dir)


    def save(self,dir,episode) :
        self.saver.save(self.sess,dir + '/critic/model.ckpt',global_step=episode)

    def clip_gradients_global_norm(self, loss, clip_factor, step):
        optimizer = tf.train.AdamOptimizer(learning_rate=step)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        filtered_grads = []
        filtered_vars = []
        for i in range(len(gradients)):
            if gradients[i] is not None:
                filtered_grads.append(gradients[i])
                filtered_vars.append(variables[i])
        gradients = filtered_grads
        variables = filtered_vars
        gradients, _ = tf.clip_by_global_norm(gradients, clip_factor) 
        grad_norm = tf.reduce_sum([tf.norm(grad) for grad in gradients])
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op

    def build_critic_network(self) :

        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32,name='state')
        self.reward = tf.placeholder(shape=[None,1],dtype=tf.float32,name='reward')

        self.outputs = self.build_base_network(self.state,1,None,'outputs',use_noise=False)

        loss = tf.losses.mean_squared_error(labels=self.reward,predictions=self.outputs)
        #optimizer_train = tf.train.AdamOptimizer(learning_rate=Config.critic_learning_rate)
        #optimizer_train = tf.contrib.estimator.clip_gradients_by_norm(optimizer_train, clip_norm=10.0)
        #self.optimizer = optimizer_train.minimize(loss)
        self.optimizer = self.clip_gradients_global_norm(loss,10.0,Config.critic_learning_rate)

    def build_base_network(self,x,output_size,output_activation,output_name=None,l2=None,use_noise=False) :
        if(Config.use_pixels) :
            if(Config.network_type == 'mlp') :
                return models.create_network_pixels_mlp(x,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name,use_noise,l2)
            elif(Config.network_type == 'conv2d') :
                return models.create_network_pixels_conv(x,Config.conv_layers,Config.conv_units,tf.nn.relu,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name,use_noise,l2)
            elif(Config.network_type == 'lstm')  :
                return models.create_network_lstm(x,Config.lstm_layers,Config.lstm_units,Config.unit_type,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name,use_noise,l2)
        elif(Config.network_type == 'mlp') :
            return models.create_mlp_network(x,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name,use_noise,l2)
        elif(Config.network_type == 'lstm') :
            return models.create_network_lstm(x,Config.lstm_layers,Config.lstm_units,Config.unit_type,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name,use_noise,l2)
        else :
            raise Exception('Unable to create base network,check config')


    def predict(self,state) :
        return self.sess.run(self.outputs,feed_dict={self.state:state})

    def train(self,states,rewards) :
        randomize = np.arange(len(states))
        for _ in range(Config.epochs) :
            if(Config.use_shuffle and Config.network_type != 'lstm') :
                np.random.shuffle(randomize)
            for index in range(int(Config.buffer_size/Config.batch_size)) :
                
                batch_states,batch_rewards = self.prepare_batch(states,rewards,index,randomize)
                self.sess.run(self.optimizer,feed_dict={self.state:batch_states,self.reward:batch_rewards})
        

    def prepare_batch(self,states,values,current_batch,randomized) :
        random_states = states[randomized].copy()
        random_values = values[randomized].copy()
        
        current_index = int(current_batch * Config.batch_size)
        batch_states = random_states[current_index : current_index + Config.batch_size]
        batch_values = random_values[current_index : current_index + Config.batch_size]

        return batch_states,batch_values