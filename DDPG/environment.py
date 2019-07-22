class Environment :

    def render(self) :
        pass

    def initialize(self) :
        pass

    def clone(self) :
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

    def actionProcessor(self,action) :
        pass

    def close(self) :
        pass