import tensorflow as tf
from config import Config
import models
import numpy as np

class ActorCritic() :
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
                self.build_network()

    def init(self,writer) :
        self.writer = writer
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        if(Config.start_timestep > 0) :
            self.load()

    def load(self) :
        dir = Config.root_dir + '/models/timestep-' + str(Config.start_timestep) + '/actor-critic-' + self.scope + '/model.ckpt-' + str(Config.start_timestep)
        
        self.saver.restore(self.sess,dir)

    def save(self,dir,timestep) :
        self.saver.save(self.sess,dir + '/actor-critic-' + self.scope + '/model.ckpt',global_step=timestep)


    def build_network(self) :
        if(self.env.is_discrete) :
            self.build_network_discrete()
        else :
            self.build_network_continuous()

    def build_network_continuous(self) :
        self.advantage = tf.placeholder(shape=[None,1],dtype=tf.float32,name='advantage')
        self.old_probs = tf.placeholder(shape=[None,self.env.get_output_size()],dtype=tf.float32,name='old_prob')
        self.reward = tf.placeholder(shape=[None,1],dtype=tf.float32,name='reward')
        self.old_preds = tf.placeholder(shape=[None,1],dtype=tf.float32,name='old_preds')
        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32,name='state')
        self.chosen_action = tf.placeholder(shape=[None,self.env.get_output_size()],dtype=tf.float32,name='chosen_action')
        
        
        base_network = self.build_base_network(self.state)
        mu_1 = tf.layers.dense(base_network,units=self.env.get_output_size(),activation=tf.nn.tanh,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        
        high = Config.env.env.action_space.high
        low = Config.env.env.action_space.low
        mu = (mu_1 * (high - low) + high + low) / 2
        
        log_sigma = tf.Variable(-0.5 * np.ones(self.env.get_output_size(), dtype=np.float32))
        sigma = tf.exp(log_sigma)

        dist = tf.contrib.distributions.Normal(mu,sigma)
        self.action = tf.clip_by_value(dist.sample(),low,high,name='action')
        self.probs = dist.prob(self.chosen_action,name='probs')

        self.value_outputs = tf.layers.dense(base_network,units=1,activation=None,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)),name='value_outputs')

        self.loss_continuous()

    def build_base_network(self,x,output_size=None,output_activation=None,output_name=None) :
        if(Config.use_pixels) :
            if(Config.network_type == 'mlp') :
                return models.create_network_pixels_mlp(x,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name)
            elif(Config.network_type == 'conv2d') :
                return models.create_network_pixels_conv(x,Config.conv_layers,Config.conv_units,Config.kernel_size,Config.strides,tf.nn.relu,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.relu,output_size,output_activation,output_name)
            elif(Config.network_type == 'lstm')  :
                return models.create_network_lstm(x,Config.lstm_layers,Config.lstm_units,Config.unit_type,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name)
        elif(Config.network_type == 'mlp') :
            return models.create_mlp_network(x,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name)
        elif(Config.network_type == 'lstm') :
            return models.create_network_lstm(x,Config.lstm_layers,Config.lstm_units,Config.unit_type,Config.mlp_hidden_layers,Config.mlp_hidden_units,tf.nn.tanh,output_size,output_activation,output_name)
        else :
            raise Exception('Unable to create base network,check config')

    def clip_by_global_norm(self,loss,optimizer,clip) :
        params = tf.trainable_variables(self.scope)
        gradients, variables = zip(*optimizer.compute_gradients(loss,params))
        gradients, _ = tf.clip_by_global_norm(gradients, clip)
        optimize = optimizer.apply_gradients(zip(gradients, variables))
        return optimize

    def loss_continuous(self) :
        ratio = self.probs / tf.maximum(1e-10,self.old_probs)
        policy_unclipped = ratio * self.advantage
        policy_clipped = tf.clip_by_value(ratio,1-Config.epsilon,1+Config.epsilon) * self.advantage
        policy_loss = tf.reduce_mean(tf.minimum(policy_unclipped,policy_clipped))
        
        value_clipped = self.old_preds + tf.clip_by_value(self.value_outputs - self.old_preds, - Config.epsilon, Config.epsilon)
        value_loss_unclipped = tf.square(self.value_outputs - self.reward)
        value_loss_clipped = tf.square(value_clipped - self.reward)
        
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_unclipped, value_loss_clipped))
        
        entropy = -tf.reduce_sum(self.probs * tf.log(tf.maximum(self.probs,1e-10)), 1)

        loss = -policy_loss - Config.entropy * entropy + value_loss * 0.5

        optimizer = tf.train.AdamOptimizer(learning_rate=Config.actor_learning_rate)
        self.optimizer = self.clip_by_global_norm(loss,optimizer,Config.gradient_clip)

    def loss_discrete(self) :
        log_prob = self.neg_log_prob
        old_log_prob = self.old_probs
        ratio = tf.exp(old_log_prob - log_prob)
        policy_unclipped = -ratio * self.advantage
        policy_clipped = -tf.clip_by_value(ratio,1-Config.epsilon,1+Config.epsilon) * self.advantage
        self.policy_loss = tf.reduce_mean(tf.maximum(policy_unclipped,policy_clipped))
        
        self.policy_loss_summary = tf.summary.scalar(name='policy_loss', tensor=self.policy_loss)

        value_clipped = self.old_preds + tf.clip_by_value(self.value_outputs - self.old_preds, - Config.epsilon, Config.epsilon)
        value_loss_unclipped = tf.square(self.value_outputs - self.reward)
        value_loss_clipped = tf.square(value_clipped - self.reward)
        
        self.value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_unclipped, value_loss_clipped))
        
        self.value_loss_summary = tf.summary.scalar(name='value_loss', tensor=self.value_loss)

        a0 = self.action_outputs - tf.reduce_max(self.action_outputs, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        self.entropy = tf.reduce_mean(tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1))

        self.entropy_summary = tf.summary.scalar(name='entropy', tensor=self.entropy)

        loss = self.policy_loss - Config.entropy * self.entropy + self.value_loss * Config.vf_factor

        self.loss_summary = tf.summary.scalar(name='loss', tensor=loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.optimizer = self.clip_by_global_norm(loss,optimizer,Config.gradient_clip)

    def build_network_discrete(self) :
        self.lr = tf.placeholder(shape=[],dtype=tf.float32,name='lr')
        self.action = tf.placeholder(shape=[None],dtype=tf.int64,name='action')
        self.advantage = tf.placeholder(shape=[None,1],dtype=tf.float32,name='advantage')
        self.reward = tf.placeholder(shape=[None,1],dtype=tf.float32,name='reward')
        self.old_preds = tf.placeholder(shape=[None,1],dtype=tf.float32,name='old_preds')
        self.old_probs = tf.placeholder(shape=[None,1],dtype=tf.float32,name='old_probs')
        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32,name='state')
        
        base_network = self.build_base_network(self.state)
        self.action_outputs = tf.layers.dense(base_network,units=self.env.get_output_size(),activation=None,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)),name='action_outputs')
        u = tf.random_uniform(tf.shape(self.action_outputs), dtype=self.action_outputs.dtype)
        self.sample = tf.argmax(self.action_outputs - tf.log(-tf.log(u)), axis=-1)
        one_hot_action = tf.one_hot(self.action, self.action_outputs.get_shape().as_list()[-1])
        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.action_outputs,labels=one_hot_action)
        self.value_outputs = tf.layers.dense(base_network,units=1,activation=None,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)),name='value_outputs')


        self.loss_discrete()

    def predict_value(self,state) :
        return self.sess.run(self.value_outputs,feed_dict={self.state:state})


    def predict_action(self,state) :
        if(self.env.is_discrete) :
            action = self.sess.run(self.sample,feed_dict={self.state:state})
            neg_log_prob = self.sess.run(self.neg_log_prob,feed_dict={self.state:state,self.action:action})
            return neg_log_prob,action
        else :
            action = self.sess.run(self.action,feed_dict={self.state:state})
            action_probs = self.sess.run(self.probs,feed_dict={self.state:state,self.chosen_action:action})
            return action,action_probs

    def train(self,states,advantages,old_probs,actions,rewards,old_preds,lr,episode) :

        randomize = np.arange(len(states))
        for epoch in range(Config.epochs) :
            if(Config.use_shuffle and Config.network_type != 'lstm') :
                np.random.shuffle(randomize)
            for index in range(int(Config.buffer_size/Config.batch_size)) :
                
                batch_states,batch_advantages,batch_old_probs,batch_actions,batch_rewards,batch_old_preds = self.prepare_batch(states,advantages,old_probs,actions,rewards,old_preds,index,randomize)
                if(self.env.is_discrete) :
                    policy_loss_t,value_loss_t,entropy_t,loss,_ = self.__train_discrete(batch_states,batch_advantages,batch_old_probs,batch_actions,batch_rewards,batch_old_preds,lr)
                    if(episode % 5 == 0 and index == 0 and epoch == 0) :
                        self.writer.add_summary(policy_loss_t)
                        self.writer.add_summary(value_loss_t)
                        self.writer.add_summary(entropy_t)
                        self.writer.add_summary(loss)
                else :
                    policy_loss_t,value_loss_t,_ = self.__train_continuous(batch_states,batch_advantages,batch_old_probs,batch_actions,batch_rewards,batch_old_preds,lr)


    def __train_discrete(self,batch_states,batch_advantages,batch_old_probs,batch_actions,batch_rewards,batch_old_preds,lr) :
        return self.sess.run([self.policy_loss_summary,self.value_loss_summary,self.entropy_summary,self.loss_summary,self.optimizer],feed_dict={self.lr:lr,self.state:batch_states,self.advantage:batch_advantages
                    ,self.old_probs:batch_old_probs,self.action:batch_actions,self.reward:batch_rewards,self.old_preds:batch_old_preds})

    def __train_continuous(self,batch_states,batch_advantages,batch_old_probs,batch_actions,batch_rewards,batch_old_preds,lr) :
        return self.sess.run([self.policy_loss,self.value_loss,self.optimizer],feed_dict={self.lr:lr,self.state:batch_states,self.advantage:batch_advantages
                    ,self.old_probs:batch_old_probs,self.chosen_action:batch_actions,self.reward:batch_rewards,self.old_preds:batch_old_preds})

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


    def prepare_batch(self,states,advantages,old_probs,actions,rewards,old_preds,current_batch,randomize) :
        random_states = states[randomize].copy()
        random_advantages = advantages[randomize].copy()
        random_old_probs = old_probs[randomize].copy()
        random_actions = actions[randomize].copy()
        random_rewards = rewards[randomize].copy()
        random_old_preds = old_preds[randomize].copy()

        current_index = int(current_batch * Config.batch_size)
        batch_states = random_states[current_index : current_index + Config.batch_size]
        batch_advantages = random_advantages[current_index : current_index + Config.batch_size]
        batch_old_probs = random_old_probs[current_index : current_index + Config.batch_size]
        batch_actions = random_actions[current_index : current_index + Config.batch_size]
        batch_rewards = random_rewards[current_index : current_index + Config.batch_size]
        batch_old_preds = random_old_preds[current_index : current_index + Config.batch_size]

        return batch_states,batch_advantages,batch_old_probs,batch_actions,batch_rewards,batch_old_preds