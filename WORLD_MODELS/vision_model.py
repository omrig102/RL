import tensorflow as tf
import numpy as np
from config import Config

class VisionModel() :

    def __init__(self,sess) :
        self.sess = sess
        self.conv_activation = tf.nn.leaky_relu


    def build_model(self,height,width) :
        self.state = tf.placeholder(shape=[None,height,width,1],dtype=tf.float32)
        self.z,self.mean,self.log_std_dev = self.__build_encoder(self.state)
        self.decoder_output = self.__build_decoder(self.z,height,width)
        flat_output = tf.reshape(self.decoder_output, [-1, height * width])
        flat_input = tf.reshape(self.state, [-1, height * width])

        with tf.name_scope('loss'):
            decoder_loss = tf.reduce_sum(tf.squared_difference(flat_output, flat_input), 1)
            kl = -0.5 * tf.reduce_sum(1.0 + self.log_std_dev - tf.square(self.mean) - tf.exp(self.log_std_dev), 1)
            self.loss = tf.reduce_mean(decoder_loss + kl)

        optimizer = tf.train.AdamOptimizer(learning_rate=Config.vision_learning_rate)
        self.optimizer = optimizer.minimize(self.loss)

    def __build_encoder(self,state) :
        with tf.variable_scope("encoder"):
            x = state
            for _ in range(Config.vision_conv_layers) :
                x = tf.layers.conv2d(x,filters=Config.vision_conv_filters,kernel_size=4,strides=2,padding='same',activation=self.conv_activation)
            x = tf.layers.flatten(x)

            mean = tf.layers.dense(x, units=Config.vision_encoder_output_size, name='mean')
            log_std_dev = tf.layers.dense(x, units=Config.vision_encoder_output_size,name='log_std_dev')

            std_dev = tf.exp(0.5 * log_std_dev)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], Config.vision_encoder_output_size]), name='epsilon')
            z = mean + epsilon * std_dev

            return z, mean, log_std_dev

    def __build_decoder(self,z,width,height) :
        with tf.variable_scope("decoder"):
            x = z
            
            x = tf.layers.dense(z, units=Config.vision_decoder_input_size * Config.vision_decoder_input_size, activation=self.conv_activation)
            x = tf.reshape(x, [-1, Config.vision_decoder_input_size, Config.vision_decoder_input_size, 1])
            for _ in range(Config.vision_conv_layers) :
                x = tf.layers.conv2d_transpose(x, filters=Config.vision_conv_filters, kernel_size=4, strides=2, padding='same', activation=self.conv_activation)


            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=height * width, activation=None)

            x = tf.layers.dense(x, units=height * width, activation=tf.nn.sigmoid)
            decoder_output = tf.reshape(x, shape=[-1, height, width, 1])
            
            return decoder_output

    def train(self,states) :
        epochs = int(len(states) / Config.vision_batch_size)
        index = 0
        loss = 0
        for _ in range(epochs) :
            batch = states[index : index + Config.vision_batch_size]
            index += Config.vision_batch_size
            _,loss_temp = self.sess.run([self.optimizer,self.loss],feed_dict={self.state:batch})
            loss += loss_temp
            
        print('loss : {}'.format(loss))
    def predict_z(self,state) :
        return self.sess.run(self.z,feed_dict={self.state:state})

    def predict_decoder(self,state) :
        return self.sess.run(self.decoder_output,feed_dict={self.state:state})