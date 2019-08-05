class Environment :

    def __init__(self,save_video=True,save_video_interval=30,root_dir='.') :
        self.save_video = save_video
        self.save_video_interval = save_video_interval
        self.root_dir = root_dir
        self.is_discrete = False

    def render(self) :
        pass

    def initialize(self) :
        pass

    def clone(self,simulator=False) :
        pass
    
    def get_input_size(self) :
        pass

    def get_output_size(self) :
        pass

    def reset(self) :
        pass

    def step(self,action) :
        pass
    
    def preprocess(self,state) :
        pass

    def close(self) :
        pass