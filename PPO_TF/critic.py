import tensorflow as tf
from config import Config
import numpy as np

class Critic() :

    def __init__(self,sess,input_size,output_size,use_pixels) :
        self.input_size = input_size
        self.output_size = output_size
        self.use_pixels = use_pixels
        self.buildCriticNetwork()
        self.sess = sess

    def buildCriticNetwork(self) :
        self.critic_input = tf.placeholder(shape=self.input_size,dtype=tf.float32)
        self.critic_value = tf.placeholder(shape=[None,1],dtype=tf.float32)
        current_layer = self.critic_input
        if(self.use_pixels and Config.use_conv_layers) :
            current_layer = tf.layers.conv2d(current_layer,filters=48,kernel_size=3,strides=1,activation=tf.nn.tanh,kernel_initializer=tf.glorot_uniform_initializer)
            current_layer = tf.layers.max_pooling2d(current_layer,pool_size=2,strides=2)
            current_layer = tf.layers.conv2d(current_layer,filters=48,kernel_size=3,strides=1,activation=tf.nn.tanh,kernel_initializer=tf.glorot_uniform_initializer)
            current_layer = tf.layers.max_pooling2d(current_layer,pool_size=2,strides=2)
            current_layer = tf.layers.flatten(current_layer)
        elif(self.use_pixels) :
            current_layer = tf.layers.reshape(current_layer,shape=[None,self.input_size[0] * self.input_size[1]])
        for _ in range(Config.hidden_size) :
            current_layer = tf.layers.dense(current_layer,units=Config.hidden_units,activation=tf.nn.relu)

        self.critic_output = tf.layers.dense(current_layer,units=1,activation=None)

        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.critic_value,predictions=self.critic_output))
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.critic_learning_rate)
        self.critic_optimizer = optimizer.minimize(loss)

    def predict(self,state) :
        return self.sess.run(self.critic_output,feed_dict={self.critic_input:state})

    def train(self,state,value) :
        self.sess.run(self.critic_optimizer,feed_dict={self.critic_input:state,self.critic_value:value})

    def prepareBatch(self,states,values,current_batch,shuffle=True) :
        current_index = int(current_batch * Config.batch_size)
        batch_states = states[current_index : current_index + Config.batch_size]
        batch_values = values[current_index : current_index + Config.batch_size]

        if(shuffle) :
            np.random.shuffle(batch_states)
            np.random.shuffle(batch_values)

        return batch_states,batch_values
