#from environment import Environment
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT,SIMPLE_MOVEMENT
import cv2
import numpy as np
from gym.wrappers import Monitor
from gym.spaces import Box
from gym_environment import GymEnvironment

class MarioEnvironment(GymEnvironment) :

    def __init__(self,game,resized_height=0,resized_width=0,use_pixels=True,stack_size=1,is_discrete=False,save_video=True,save_video_interval=30,root_dir='.') :
        super().__init__(game,resized_height,resized_width,use_pixels,stack_size,is_discrete,save_video,save_video_interval,root_dir)

    def initialize(self) :
        self.env = gym_super_mario_bros.make(self.game)
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)
        if(self.save_video) :
            self.env = Monitor(self.env,self.root_dir + "/videos",video_callable=lambda episode_id : True if(episode_id % self.save_video_interval == 0) else False,force=True)

    def clone(self,simulator=False) :
        if(simulator) :
            return MarioEnvironment(self.game,self.resized_height,self.resized_width,self.use_pixels,self.stack_size,self.is_discrete,False,self.save_video_interval,self.root_dir)
        return MarioEnvironment(self.game,self.resized_height,self.resized_width,self.use_pixels,self.stack_size,self.is_discrete,self.save_video,self.save_video_interval,self.root_dir)