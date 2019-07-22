from environment import Environment
import gym
import cv2
import numpy as np
from gym.spaces import Box

class GymEnvironment(Environment) :

    def __init__(self,game,resized_height=0,resized_width=0,use_pixels=True) :
        self.game = game
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.use_pixels = use_pixels
        self.input_size = None
        self.output_size = None
        
        
    def render(self) :
        self.env.render()

    def initialize(self) :
        self.env = gym.make(self.game)

    def clone(self) :
        return GymEnvironment(self.game,self.resized_height,self.resized_width,self.use_pixels)

    def getInputSize(self) :
        if(self.input_size is None) :
            if(self.use_pixels) :
                self.input_size = self.resized_height * self.resized_width
            else :
                self.initialize()
                self.input_size = self.env.observation_space.shape[0]
                self.close()

        return self.input_size

    def getOutputSize(self) :
        if(self.output_size is None) :
            self.initialize()
            if(type(self.env.action_space) == Box) :
                self.output_size = self.env.action_space.shape[0]
            else :
                self.output_size = self.env.action_space.n
            self.close()
        return self.output_size

    def reset(self) :
        return self.env.reset()

    def step(self,action) :
        return self.env.step(action)

    def preprocess(self,state) :
        if(self.use_pixels) :
            frame = cv2.resize(state,(self.resized_width,self.resized_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame.reshape([self.resized_width * self.resized_height])
        return state

    def actionProcessor(self,action) :
        #if(type(self.env.action_space) == Box) :
        #    return np.argmax(action)
        return np.reshape(action,newshape=[action.shape[1]])

    def close(self) :
        self.env.close()