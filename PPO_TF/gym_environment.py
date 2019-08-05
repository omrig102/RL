from environment import Environment
import gym
import cv2
import numpy as np
from gym.spaces import Box
from gym.wrappers import Monitor

class GymEnvironment(Environment) :

    def __init__(self,game,save_video=True,save_video_interval=50,root_dir='.') :
        super().__init__(save_video,save_video_interval,root_dir)
        self.game = game
        self.input_size = None
        self.output_size = None
        self.initialize()
        self.get_input_size()
        self.get_output_size()

    def render(self) :
        self.env.render()

    def initialize(self) :
        self.env = gym.make(self.game)
        if(self.save_video) :
            self.env = Monitor(self.env,self.root_dir + "/videos",video_callable=lambda episode_id : True if(episode_id % self.save_video_interval == 0) else False,force=True)

    def clone(self,simulator=False) :
        return GymEnvironment(self.game,not simulator,self.save_video_interval,self.root_dir)

    def get_input_size(self) :
        if(self.input_size is None) :
            if(type(self.env.observation_space) == Box) :
                self.input_size = self.env.observation_space.shape
            else :
                self.input_size = self.env.observation_space.n
        return self.input_size

    def get_output_size(self) :
        if(self.output_size is None) :
            if(type(self.env.action_space) == Box) :
                self.output_size = self.env.action_space.shape[0]
                self.is_discrete = False
            else :
                self.output_size = self.env.action_space.n
                self.is_discrete = True
        return self.output_size

    def reset(self) :
        return self.env.reset()

    def step(self,action) :
        return self.env.step(action)



    def close(self) :
        self.env.close()
