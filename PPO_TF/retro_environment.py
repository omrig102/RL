from gym_environment import GymEnvironment
import retro
from gym.wrappers import Monitor

class RetroEnvironment(GymEnvironment) :

    def __init__(self,game,save_video=True,save_video_interval=30,root_dir='.') :
        super().__init__(game,save_video,save_video_interval,root_dir)



    def initialize(self) :
        self.env = retro.make(game=self.game,use_restricted_actions=retro.Actions.DISCRETE)
        if(self.save_video) :
            self.env = Monitor(self.env,self.root_dir + "/videos",video_callable=lambda episode_id : True if(episode_id % self.save_video_interval == 0) else False,force=True)

    def clone(self,simulator=False) :
        save_video = True
        if(simulator) :
            save_video = False
        else :
            save_video = self.save_video

        return RetroEnvironment(self.game,save_video,self.save_video_interval,self.root_dir)
