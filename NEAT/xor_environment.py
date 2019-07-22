from environment import Environment

class XOREnvironment(Environment) :

    def __init__(self) :
        self.states = [[0,0],[0,1],[1,0],[1,1]]
        self.current_state_index = 0

    def render(self) :
        pass

    def initialize(self) :
        pass

    def clone(self) :
        return XOREnvironment()

    def getInputSize(self) :
        return 2

    def getOutputSize(self) :
        return 1

    def reset(self) :
        self.current_state_index = 0
        return self.states[self.current_state_index]
        
    def step(self,action) :
        current_state = self.states[self.current_state_index]
        if(current_state[0] == 0) :
            if(current_state[1] == 0) :
                expected = 0
            else :
                expected = 1
        else :
            if(current_state[1] == 0) :
                expected = 1
            else :
                expected = 0

        fitness = 1 - abs(expected - action)
        self.current_state_index += 1
        if(self.current_state_index >= len(self.states)) :
            self.current_state_index = 0
        next_state = self.states[self.current_state_index]

        return (next_state,fitness,False,None)
    
    def preprocess(self,state) :
        return state

    def actionProcessor(self,action) :
        return action[0]    

    def close(self) :
        pass