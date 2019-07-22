from config import Config
from node import Node

class Connection() :

    def __init__(self,input,output,innovation,weight,enabled) :
        self.input = input
        self.output = output
        self.innovation = innovation
        self.weight = weight
        self.enabled = enabled


    def clone(self) :
        new_input = self.input.clone()
        new_output = self.output.clone()
        return Connection(new_input,new_output,self.innovation,self.weight,self.enabled)