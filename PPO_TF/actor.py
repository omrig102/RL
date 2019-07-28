import tensorflow as tf
from config import Config
import numpy as np

class Actor() :

    def __init__(self,sess,input_size,output_size,use_pixels,is_discrete,scope) :
        self.input_size = input_size
        self.output_size = output_size
        self.use_pixels = use_pixels
        self.is_discrete = is_discrete
        self.scope = scope
        with tf.variable_scope(scope) as s:
            self.buildActorNetwork()
        self.sess = sess

    def buildActorNetwork(self) :
        if(self.is_discrete) :
            self.buildActorNetworkDiscrete()
        else :
            self.buildActorNetworkContinuous()
    
    def get_gaussian_log(self,x, mu, log_stddev):
        '''
        returns log probability of picking x
        from a gaussian distribution N(mu, stddev)
        '''
        # ignore constant since it will be cancelled while taking ratios
        log_prob = -log_stddev - (x - mu)**2 / (2 * tf.exp(log_stddev)**2)
        return log_prob

    def buildActorNetworkContinuous(self) :
        self.mask = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32)
        self.advantage = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.old_probs = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32)
        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32)
        self.actor_action_p = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32)
        current_layer = self.state
        if(self.use_pixels and Config.use_conv_layers) :
            current_layer = tf.layers.conv2d(current_layer,filters=48,kernel_size=3,strides=1,activation=tf.nn.relu)
            current_layer = tf.layers.conv2d(current_layer,filters=48,kernel_size=3,strides=1,activation=tf.nn.relu)
            current_layer = tf.layers.flatten(current_layer)
        elif(self.use_pixels) :
            current_layer = tf.reshape(current_layer,shape=[-1,self.input_size[1] * self.input_size[2] * self.input_size[3]])

        for _ in range(Config.hidden_size) :
            current_layer = tf.layers.dense(current_layer,units=Config.hidden_units,activation=tf.nn.tanh)
        
        #current_layer = tf.layers.batch_normalization(current_layer, training=True)
        mu_1 = tf.layers.dense(current_layer,units=self.output_size,activation=tf.nn.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)) 
        
        mu = mu_1 * (Config.env.env.action_space.high - Config.env.env.action_space.low ) / 2
        
        log_sigma = tf.Variable(np.zeros(self.output_size, dtype=np.float32))
        sigma = tf.exp(log_sigma)

        
        dist = tf.contrib.distributions.Normal(mu,sigma)
        self.actor_action = tf.clip_by_value(dist.sample(1),Config.env.env.action_space.low,Config.env.env.action_space.high)
        self.actor_probs = dist.prob(self.actor_action_p)

        ratio = self.actor_probs / (self.old_probs + 1e-10)
        unclipped = ratio * self.advantage
        clipped = tf.clip_by_value(ratio,1-Config.epsilon,1+Config.epsilon) * self.advantage
        loss = -tf.reduce_mean(tf.minimum(unclipped,clipped) + Config.entropy * -(self.actor_probs * tf.log(self.actor_probs + 1e-10)))
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.actor_learning_rate)
        self.actor_optimizer = optimizer.minimize(loss)

    def buildActorNetworkDiscrete(self) :
        self.mask = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32)
        self.advantage = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.old_probs = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32)
        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32)
        
        current_layer = self.state
        if(self.use_pixels and Config.use_conv_layers) :
            current_layer = tf.layers.conv2d(current_layer,filters=48,kernel_size=3,strides=1,activation=tf.nn.relu)
            current_layer = tf.layers.conv2d(current_layer,filters=48,kernel_size=3,strides=1,activation=tf.nn.relu)
            current_layer = tf.layers.flatten(current_layer)
        elif(self.use_pixels) :
            current_layer = tf.layers.reshape(current_layer,shape=[-1,self.input_size[1] * self.input_size[2] * self.input_size[3]])
        for _ in range(Config.hidden_size) :
            current_layer = tf.layers.dense(current_layer,units=Config.hidden_units,activation=tf.nn.relu)

        self.actor_outputs = tf.layers.dense(current_layer,units=self.output_size,activation=tf.nn.softmax)

        prob = tf.log(self.mask * self.actor_outputs + 1e-10)
        old_prob = tf.log(self.mask * self.old_probs + 1e-10)
        ratio = tf.exp(prob - old_prob)
        unclipped = ratio * self.advantage
        clipped = tf.clip_by_value(ratio,1-Config.epsilon,1+Config.epsilon) * self.advantage
        loss = -tf.reduce_mean(tf.minimum(unclipped,clipped) + Config.entropy * -(self.mask * self.actor_outputs * prob))
        #loss = ppoLoss(advantage=self.advantage,old_actions_probs=self.old_probs)
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.actor_learning_rate)
        self.actor_optimizer = optimizer.minimize(loss)


    def predict(self,state) :
        if(self.is_discrete) :
            return self.sess.run(self.actor_outputs,feed_dict={self.state:state})
        else :
            action = self.sess.run(self.actor_action,feed_dict={self.state:state})
            action = action[0]
            action_probs = self.sess.run(self.actor_probs,feed_dict={self.state:state,self.actor_action_p:action})
            return action,action_probs

    def train(self,state,advantage,old_probs,mask) :
        if(self.is_discrete) :
            self.sess.run(self.actor_optimizer,feed_dict={self.state:state,self.advantage:advantage
            ,self.old_probs:old_probs,self.mask:mask})
        else :
            self.sess.run(self.actor_optimizer,feed_dict={self.state:state,self.advantage:advantage
            ,self.old_probs:old_probs,self.mask:mask,self.actor_action_p:mask})

    def copyTrainables(self,actor_scope) :
        e1_params = [t for t in tf.trainable_variables(actor_scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables(self.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)
        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        self.sess.run(update_ops)


    def prepareBatch(self,states,advantages,old_probs,masks,current_batch,randomize) :
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

