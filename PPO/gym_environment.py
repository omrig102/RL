from environment import Environment
import gym
import cv2
import numpy as np
from gym.spaces import Box
from gym.wrappers import Monitor

class GymEnvironment(Environment) :

    def __init__(self,game,resized_height=0,resized_width=0,use_pixels=True,stack_size=1,is_discrete=True,save_video=True,save_video_interval=50,root_dir='.') :
        super().__init__(game,resized_height,resized_width,use_pixels,stack_size,is_discrete,save_video,save_video_interval,root_dir)
        
    def render(self) :
        self.env.render()

    def initialize(self) :
        self.env = gym.make(self.game)
        if(self.save_video) :
            self.env = Monitor(self.env,self.root_dir + "/videos",video_callable=lambda episode_id : True if(episode_id % self.save_video_interval == 0) else False,force=True)

    def clone(self,simulator=False) :
        if(simulator) :
            return GymEnvironment(self.game,self.resized_height,self.resized_width,self.use_pixels,self.stack_size,self.is_discrete,False,self.save_video_interval,self.root_dir)
        return GymEnvironment(self.game,self.resized_height,self.resized_width,self.use_pixels,self.stack_size,self.is_discrete,self.save_video,self.save_video_interval,self.root_dir)

    def getInputSize(self) :
        if(self.input_size is None) :
            if(self.use_pixels) :
                self.input_size = [self.resized_height, self.resized_width,self.stack_size]
            else :
                self.input_size = [self.env.observation_space.shape[0]]

        return self.input_size

    def getOutputSize(self) :
        if(self.output_size is None) :
            if(type(self.env.action_space) == Box) :
                self.output_size = self.env.action_space.shape[0]
            else :
                self.output_size = self.env.action_space.n
        return self.output_size

    def reset(self) :
        return self.env.reset()

    def step(self,action) :
        return self.env.step(action)

    def preprocess(self,state,next_state) :
        if(self.use_pixels) :
            if(self.stack_size > 1) :
                if(next_state is not None) :
                    frame = cv2.resize(next_state,(self.resized_width,self.resized_height))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
                    frame = frame.reshape([self.resized_width, self.resized_height])
                    stack = state[:,:,1:]
                    res = []
                    for index in range(self.stack_size) :
                        if(index == self.stack_size - 1) :
                            res.append(frame)
                        else :
                            res.append(stack[:,:,index])
                    return np.stack(res,axis=2)
                else :
                    frame = cv2.resize(state,(self.resized_width,self.resized_height))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
                    frame = frame.reshape([self.resized_width, self.resized_height])
                    return np.stack([frame for _ in range(self.stack_size)],axis=2)
            else :
                if(next_state is not None) :
                    state = next_state
                frame = cv2.resize(state,(self.resized_width,self.resized_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
                return frame.reshape([self.resized_width, self.resized_height, 1])
        if(next_state is None) :
            return state
        return next_state

    def close(self) :
        self.env.close()
