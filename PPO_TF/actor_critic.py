from config import Config
import tensorflow as tf
import models
import numpy as np

class ActorCritic() :

    def __init__(self,sess,env,scope) :
        self.sess = sess
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
        with tf.variable_scope(scope) as s:
            self.create_network()


    def create_network(self) :
        if(self.env.is_discrete) :
            self.create_network_discrete()
        else :
            self.create_network_continuous()

    def create_network_discrete(self) :
        self.mask = tf.placeholder(shape=[None,self.env.get_output_size()],dtype=tf.float32,name='mask')
        self.advantage = tf.placeholder(shape=[None,1],dtype=tf.float32,name='advantage')
        self.old_probs = tf.placeholder(shape=[None,self.env.get_output_size()],dtype=tf.float32,name='old_probs')
        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32,name='state')
        self.rewards = tf.placeholder(shape=[None,1],dtype=tf.float32,name='rewards')

        x = self.build_base_network(self.state,l2=Config.l2)

        reg = tf.contrib.layers.l2_regularizer(Config.l2)
        self.actor_outputs = tf.layers.dense(x,units=self.env.get_output_size(),activation=tf.nn.softmax,kernel_regularizer=reg)
        self.critic_outputs = tf.layers.dense(x,units=1,activation=None)

        critic_loss = tf.square(self.critic_outputs - self.rewards)
        prob = tf.log(self.mask * self.actor_outputs + 1e-10)
        old_prob = tf.log(self.mask * self.old_probs + 1e-10)
        ratio = tf.exp(prob - old_prob)
        unclipped = ratio * self.advantage
        clipped = tf.clip_by_value(ratio,1-Config.epsilon,1+Config.epsilon) * self.advantage
        actor_loss = -(tf.minimum(unclipped,clipped))

        loss = tf.reduce_mean(actor_loss + 1.0 * critic_loss + Config.entropy * -tf.reduce_sum(self.mask * self.actor_outputs * prob))

        optimizer = tf.train.AdamOptimizer(learning_rate=Config.actor_learning_rate)
        self.optimizer = optimizer.minimize(loss)

    def build_base_network(self,x,l2=None) :
        if(Config.use_pixels) :
            if(Config.network_type == 'mlp') :
                return models.create_network_pixels_mlp(x,Config.mlp_hidden_layers,Config.mlp_hidden_units,'tanh',l2=l2)
            elif(Config.network_type == 'conv2d') :
                return models.create_network_pixels_conv(x,Config.conv_layers,Config.conv_units,'relu',Config.mlp_hidden_layers,Config.mlp_hidden_units,'tanh',l2=l2)
            elif(Config.network_type == 'lstm')  :
                return models.create_network_lstm(x,Config.lstm_layers,Config.lstm_units,Config.unit_type,Config.mlp_hidden_layers,Config.mlp_hidden_units,'tanh',l2=l2)
        elif(Config.network_type == 'mlp') :
            return models.create_mlp_network(x,Config.mlp_hidden_layers,Config.mlp_hidden_units,'tanh',l2=l2)
        elif(Config.network_type == 'lstm') :
            return models.create_network_lstm(x,Config.lstm_layers,Config.lstm_units,Config.unit_type,Config.mlp_hidden_layers,Config.mlp_hidden_units,'tanh',l2=l2)
        else :
            raise Exception('Unable to create base network,check config')

    def create_network_continuous(self) :
        pass

    def predict_actor(self,state) :
        if(self.env.is_discrete) :
            return self.sess.run(self.actor_outputs,feed_dict={self.state:state})
        else :
            action = self.sess.run(self.action,feed_dict={self.state:state})
            action = action[0]
            action_probs = self.sess.run(self.probs,feed_dict={self.state:state,self.chosen_action:action})
            return action,action_probs
    
    def predict_critic(self,state) :
        return self.sess.run(self.critic_outputs,feed_dict={self.state:state})

    def train(self,states,advantages,old_probs,masks,rewards) :
        randomize = np.arange(len(states))
        for _ in range(Config.epochs) :
            #self.sess.run(iterator.initializer)
            for index in range(int(Config.buffer_size/Config.batch_size)) :
                if(Config.use_shuffle and Config.network_type != 'lstm') :
                    np.random.shuffle(randomize)
                batch_states,batch_advantages,batch_old_probs,batch_masks,batch_rewards = self.prepare_batch(states,advantages,old_probs,masks,rewards,index,randomize)
                #batch_states,batch_advantages,batch_old_probs,batch_masks = self.sess.run(batch)
                if(self.env.is_discrete) :
                    self.__train_discrete(batch_states,batch_advantages,batch_old_probs,batch_masks,batch_rewards)
                else :
                    self.__train_continuous(batch_states,batch_advantages,batch_old_probs,batch_masks)

    def __train_discrete(self,batch_states,batch_advantages,batch_old_probs,batch_masks,batch_rewards) :
        self.sess.run(self.optimizer,feed_dict={self.state:batch_states,self.advantage:batch_advantages
                    ,self.old_probs:batch_old_probs,self.mask:batch_masks,self.rewards:batch_rewards})

    def __train_continuous(self,batch_states,batch_advantages,batch_old_probs,batch_actions) :
        self.sess.run(self.optimizer,feed_dict={self.state:batch_states,self.advantage:batch_advantages
                    ,self.old_probs:batch_old_probs,self.chosen_action:batch_actions})

    def prepare_batch(self,states,advantages,old_probs,masks,rewards,current_batch,randomize) :
        random_states = states[randomize].copy()
        random_advantages = advantages[randomize].copy()
        random_old_probs = old_probs[randomize].copy()
        random_masks = masks[randomize].copy()
        random_rewards = rewards[randomize].copy()

        current_index = int(current_batch * Config.batch_size)
        batch_states = random_states[current_index : current_index + Config.batch_size]
        batch_advantages = random_advantages[current_index : current_index + Config.batch_size]
        batch_old_probs = random_old_probs[current_index : current_index + Config.batch_size]
        batch_masks = random_masks[current_index : current_index + Config.batch_size]
        batch_rewards = random_rewards[current_index : current_index + Config.batch_size]

        return batch_states,batch_advantages,batch_old_probs,batch_masks,batch_rewards

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