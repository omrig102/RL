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

    def __init__(self,game,save_video=True,save_video_interval=30,root_dir='.') :
        super().__init__(game,save_video,save_video_interval,root_dir)

    def initialize(self) :
        self.env = gym_super_mario_bros.make(self.game)
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)
        if(self.save_video) :
            self.env = Monitor(self.env,self.root_dir + "/videos",video_callable=lambda episode_id : True if(episode_id % self.save_video_interval == 0) else False,force=True)

    def clone(self,simulator=False) :
        return MarioEnvironment(self.game,not simulator,self.save_video_interval,self.root_dir)
