import tensorflow as tf
from config import Config
import models
import numpy as np

class ActorCritic() :
    def __init__(self,sess,state_size,action_size,is_discrete,scope) :
        self.scope = scope
        self.input_size = [None]
        self.output_size = action_size
        self.is_discrete = is_discrete
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
            for shape in state_size :
                self.input_size.append(shape)
        
        self.sess = sess
        with tf.variable_scope(scope) as s:
                self.build_network()

    def init(self,writer=None) :
        self.writer = writer
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))
        if(Config.start_timestep > 0) :
            self.load()

    def load(self) :
        dir = Config.root_dir + '/models/timestep-' + str(Config.start_timestep) + '/actor-critic-' + self.scope + '/model.ckpt-' + str(Config.start_timestep)
        
        self.saver.restore(self.sess,dir)

    def save(self,dir,current_timestep) :
        self.saver.save(self.sess,dir + '/actor-critic-' + self.scope + '/model.ckpt',global_step=current_timestep)


    def build_network(self) :
        if(self.is_discrete) :
            self.build_network_discrete()
            self.calc_loss(self.action_prob,self.action_old_prob)
        else :
            self.build_network_continuous()
            self.calc_loss(self.probs,self.old_probs)

    def build_network_continuous(self) :
        self.advantage = tf.placeholder(shape=[None,1],dtype=tf.float32,name='advantage')
        self.old_probs = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32,name='old_prob')
        self.reward = tf.placeholder(shape=[None,1],dtype=tf.float32,name='reward')
        self.old_preds = tf.placeholder(shape=[None,1],dtype=tf.float32,name='old_preds')
        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32,name='state')
        self.action = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32,name='action')
        
        
        base_network = self.build_base_network(self.state)
        mu_1 = tf.layers.dense(base_network,units=self.output_size,activation=tf.nn.tanh,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)))
        
        high = Config.env.env.action_space.high
        low = Config.env.env.action_space.low
        mu = (mu_1 * (high - low) + high + low) / 2
        
        log_sigma = tf.Variable(-0.5 * np.ones(self.output_size, dtype=np.float32))
        sigma = tf.exp(log_sigma)

        dist = tf.contrib.distributions.Normal(mu,sigma)
        self.action = tf.clip_by_value(dist.sample(),low,high,name='action')
        self.probs = dist.prob(self.action,name='probs')

        self.value_outputs = tf.layers.dense(base_network,units=1,activation=None,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)),name='value_outputs')

    def build_network_discrete(self) :
        self.lr = tf.placeholder(shape=[],dtype=tf.float32,name='lr')
        self.action = tf.placeholder(shape=[None],dtype=tf.int64,name='action')
        self.advantage = tf.placeholder(shape=[None,1],dtype=tf.float32,name='advantage')
        self.reward = tf.placeholder(shape=[None,1],dtype=tf.float32,name='reward')
        self.old_preds = tf.placeholder(shape=[None,1],dtype=tf.float32,name='old_preds')
        self.old_probs = tf.placeholder(shape=[None,self.output_size],dtype=tf.float32,name='old_probs')
        self.state = tf.placeholder(shape=self.input_size,dtype=tf.float32,name='state')
        
        base_network = self.build_base_network(self.state)
        self.action_outputs = tf.layers.dense(base_network,units=self.output_size,activation=tf.nn.softmax,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)),name='action_outputs')
        
        action_one_hot = tf.one_hot(self.action,self.output_size)
        self.action_prob = action_one_hot * self.action_outputs
        self.action_old_prob = action_one_hot * self.old_probs
        
        self.value_outputs = tf.layers.dense(base_network,units=1,activation=None,kernel_initializer=tf.orthogonal_initializer(np.sqrt(2)),name='value_outputs')

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

    def calc_loss(self,action_prob,action_old_prob) :
        ratio = action_prob / (tf.maximum(1e-10,action_old_prob))
        policy_unclipped = -ratio * self.advantage
        policy_clipped = -tf.clip_by_value(ratio,1-Config.epsilon,1+Config.epsilon) * self.advantage
        policy_loss = tf.reduce_mean(tf.maximum(policy_unclipped,policy_clipped))
        
        value_clipped = self.old_preds + tf.clip_by_value(self.value_outputs - self.old_preds, - Config.epsilon, Config.epsilon)
        value_loss_unclipped = tf.square(self.value_outputs - self.reward)
        value_loss_clipped = tf.square(value_clipped - self.reward)
        
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_unclipped, value_loss_clipped))
        
        entropy = -tf.reduce_mean(tf.reduce_sum(action_prob * tf.log(tf.maximum(action_prob,1e-10)), 1),0)

        loss = policy_loss - Config.entropy * entropy + value_loss * 0.5

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.optimizer = self.clip_by_global_norm(loss,optimizer,Config.gradient_clip)

        if(Config.with_summary) :
            self.policy_loss_summary = tf.summary.scalar(name='policy_loss', tensor=policy_loss)
            self.value_loss_summary = tf.summary.scalar(name='value_loss', tensor=value_loss)
            self.entropy_summary = tf.summary.scalar(name='entropy', tensor=entropy)
            self.loss_summary = tf.summary.scalar(name='loss', tensor=loss)




    def predict_value(self,state) :
        values = self.sess.run(self.value_outputs,feed_dict={self.state:state})
        return values.reshape(values.shape[0])


    def predict_action(self,state) :
        if(self.is_discrete) :
            action_probs = self.sess.run(self.action_outputs,feed_dict={self.state:state})
            action_probs = action_probs.reshape([action_probs.shape[1]])
            action = np.random.choice(range(len(action_probs)),p=action_probs)
            
        else :
            action = self.sess.run(self.action,feed_dict={self.state:state})
            action_probs = self.sess.run(self.probs,feed_dict={self.state:state,self.action:action})
            

        return action,action_probs

    def train(self,batch,current_timestep) :
        current_update = current_timestep / Config.buffer_size
        total_updates = Config.timesteps / Config.buffer_size
        frac = 1.0 - (current_update - 1.0) / total_updates
        lr = Config.actor_learning_rate(frac)
        states,rewards,actions,old_probs,advantages,old_preds = batch

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        randomize = np.arange(len(states))
        for epoch in range(Config.epochs) :
            if(Config.use_shuffle and Config.network_type != 'lstm') :
                np.random.shuffle(randomize)
            for index in range(int(Config.buffer_size/Config.batch_size)) :
                
                batch_states,batch_advantages,batch_old_probs,batch_actions,batch_rewards,batch_old_preds = self.prepare_batch(states,advantages,old_probs,actions,rewards,old_preds,index,randomize)
                if(Config.with_summary) :
                    policy_loss_t,value_loss_t,entropy_t,loss,_ = self.sess.run([self.policy_loss_summary,self.value_loss_summary,self.entropy_summary,self.loss_summary,self.optimizer],feed_dict={self.state:batch_states,self.advantage:batch_advantages
                        ,self.old_probs:batch_old_probs,self.action:batch_actions,self.reward:batch_rewards,self.old_preds:batch_old_preds,self.lr:lr})
                    if(index == int(Config.buffer_size/Config.batch_size)-1 and epoch == Config.epochs-1) :
                        self.writer.add_summary(policy_loss_t)
                        self.writer.add_summary(value_loss_t)
                        self.writer.add_summary(entropy_t)
                        self.writer.add_summary(loss)
                else :
                    self.sess.run(self.optimizer,feed_dict={self.state:batch_states,self.advantage:batch_advantages
                        ,self.old_probs:batch_old_probs,self.action:batch_actions,self.reward:batch_rewards,self.old_preds:batch_old_preds,self.lr:lr})


    def prepare_batch(self,states,advantages,old_probs,actions,rewards,old_preds,current_batch,randomize) :
        current_index = int(current_batch * Config.batch_size)
        batch_states = states[current_index : current_index + Config.batch_size]
        batch_advantages = advantages[current_index : current_index + Config.batch_size]
        batch_old_probs = old_probs[current_index : current_index + Config.batch_size]
        batch_actions = actions[current_index : current_index + Config.batch_size]
        batch_rewards = rewards[current_index : current_index + Config.batch_size]
        batch_old_preds = old_preds[current_index : current_index + Config.batch_size]

        return batch_states,batch_advantages,batch_old_probs,batch_actions,batch_rewards,batch_old_preds