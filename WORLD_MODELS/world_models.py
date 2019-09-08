from config import Config
from vision_model import VisionModel
from controller_model import ControllerModel
from memory_model import MemoryModel


class WorldModels() :

    def __init__(self) :
        self.vision_model = VisionModel()
        self.memory_model = MemoryModel()
        self.controller_model = ControllerModel()
        
        self.vision_model.build_model()
        self.memory_model.build_model()
        self.controller_model.build_model()


    def train(self) :
        self.vision_model.train()
        self.memory_model.train()
        self.controller_model.train()

    def act(self) :
        pass


    def run(self) :
        for episode in range(Config.episodes) :
            for step in range(Config.rollout_size) :
