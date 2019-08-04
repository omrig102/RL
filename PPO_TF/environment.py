class Environment :

    def __init__(self,game=None,resized_height=0,resized_width=0,use_pixels=True,use_conv=False,use_lstm=False,stack_size=1,is_discrete=False,save_video=True,save_video_interval=30,root_dir='.') :
        self.game = game
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.use_pixels = use_pixels
        self.stack_size = stack_size
        self.input_size = None
        self.output_size = None
        self.is_discrete = is_discrete
        self.save_video = save_video
        self.save_video_interval = save_video_interval
        self.root_dir = root_dir
        self.use_conv = use_conv
        self.use_lstm = use_lstm
        self.initialize()

    def render(self) :
        pass

    def initialize(self) :
        pass

    def clone(self,simulator=False) :
        pass
    
    def getInputSize(self) :
        pass

    def getOutputSize(self) :
        pass

    def reset(self) :
        pass

    def step(self,action) :
        pass
    
    def preprocess(self,state) :
        pass

    def close(self) :
        pass