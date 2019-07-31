import tensorflow as tf
from config import Config
import numpy as np

class Critic() :

    def __init__(self,sess,input_size,output_size,use_pixels,scope) :
        self.input_size = input_size
        self.output_size = output_size
        self.use_pixels = use_pixels
        with tf.variable_scope(scope) as s:
            self.buildCriticNetwork()
        self.sess = sess
        self.scope = scope

    def buildCriticNetwork(self) :
        self.critic_input = tf.placeholder(shape=self.input_size,dtype=tf.float32)
        self.critic_value = tf.placeholder(shape=[None,1],dtype=tf.float32)
        current_layer = self.critic_input
        if(self.use_pixels and Config.use_conv_layers) :
            current_layer = tf.layers.conv2d(current_layer,filters=48,kernel_size=3,strides=1,activation=tf.nn.relu)
            current_layer = tf.layers.conv2d(current_layer,filters=48,kernel_size=3,strides=1,activation=tf.nn.relu)
            current_layer = tf.layers.flatten(current_layer)
        elif(self.use_pixels) :
            current_layer = tf.reshape(current_layer,shape=[-1,self.input_size[1] * self.input_size[2] * self.input_size[3]])
        for _ in range(Config.hidden_size) :
            current_layer = tf.layers.dense(current_layer,units=Config.hidden_units,activation=tf.nn.relu)

        self.critic_output = tf.layers.dense(current_layer,units=1,activation=None)

        #ratio = self.critic_output / (self.old_outputs + 1e-10)
        loss = tf.losses.mean_squared_error(labels=self.critic_value,predictions=self.critic_output)
        #unclipped = ratio * loss
        #clipped = tf.clip_by_value(ratio,1-Config.epsilon,1+Config.epsilon) * loss
        #loss = tf.reduce_mean(tf.minimum(clipped,unclipped))
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.critic_learning_rate)
        self.critic_optimizer = optimizer.minimize(loss)

    def predict(self,state) :
        return self.sess.run(self.critic_output,feed_dict={self.critic_input:state})

    def train(self,state,value) :
        self.sess.run(self.critic_optimizer,feed_dict={self.critic_input:state,self.critic_value:value})

    def prepareBatch(self,states,values,current_batch,randomized) :
        random_states = states[randomized].copy()
        random_values = values[randomized].copy()
        
        current_index = int(current_batch * Config.batch_size)
        batch_states = random_states[current_index : current_index + Config.batch_size]
        batch_values = random_values[current_index : current_index + Config.batch_size]

        return batch_states,batch_values