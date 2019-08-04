import tensorflow as tf
from config import Config
import numpy as np
import models

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
            for shape in self.env.getInputSize() :
                self.input_size.append(shape)
        with tf.variable_scope(scope) as s:
            self.buildCriticNetwork()
        self.sess = sess
        
        self.scope = scope

    def buildCriticNetwork(self) :

        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32)
        self.reward = tf.placeholder(shape=[None,1],dtype=tf.float32)

        self.outputs = self.build_base_network(self.state,1,None)

        loss = tf.losses.mean_squared_error(labels=self.reward,predictions=self.outputs)
        optimizer_train = tf.train.AdamOptimizer(learning_rate=Config.critic_learning_rate)
        self.optimizer = optimizer_train.minimize(loss)

    def build_base_network(self,x,output_size,output_activation) :
        if(Config.use_pixels) :
            if(Config.network_type == 'mlp') :
                return models.create_network_pixels_mlp(x,Config.hidden_layers,Config.hidden_units,'tanh',output_size,output_activation)
            elif(Config.network_type == 'conv2d') :
                return models.create_network_pixels_conv(x,2,128,'relu',Config.hidden_layers,Config.hidden_units,'tanh',output_size,output_activation)
            elif(Config.network_type == 'lstm')  :
                return models.create_network_lstm(x,Config.lstm_layers,Config.lstm_units,Config.hidden_layers,Config.hidden_units,'tanh',output_size,output_activation)
        elif(Config.network_type == 'mlp') :
            return models.create_mlp_network(x,Config.hidden_layers,Config.hidden_units,'tanh',output_size,output_activation)
        elif(Config.network_type == 'lstm') :
            return models.create_network_lstm(x,Config.lstm_layers,Config.lstm_units,Config.hidden_layers,Config.hidden_units,'tanh',output_size,output_activation)
        else :
            raise Exception('Unable to create base network,check config')


    def predict(self,state) :
        return self.sess.run(self.outputs,feed_dict={self.state:state})

    def train(self,states,rewards) :
        randomize = np.arange(len(states))
        for _ in range(Config.epochs) :
            for index in range(int(Config.buffer_size/Config.batch_size)) :
                if(Config.use_shuffle and Config.network_type != 'lstm') :
                    np.random.shuffle(randomize)
                batch_states,batch_rewards = self.prepareBatch(states,rewards,index,randomize)
                self.sess.run(self.optimizer,feed_dict={self.state:batch_states,self.reward:batch_rewards})
        

    def prepareBatch(self,states,values,current_batch,randomized) :
        random_states = states[randomized].copy()
        random_values = values[randomized].copy()
        
        current_index = int(current_batch * Config.batch_size)
        batch_states = random_states[current_index : current_index + Config.batch_size]
        batch_values = random_values[current_index : current_index + Config.batch_size]

        return batch_states,batch_values
