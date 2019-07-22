from config import Config

class Node() :

    def __init__(self,type,activation,id) :
        self.type = type
        self.activation = activation
        self.id = id
        self.order = -1
        self.value = None


    def clone(self) :
        node = Node(self.type,self.activation,self.id)
        if(node.type == 'input' or node.type == 'output') :
            node.order = self.order
        return node