import math

class Activation :

    def relu(x) :
        if(x <= 0) :
            return 0

        return x

    def sigmoid(x):
        if x < 0:
            return 1 - 1/(1 + math.exp(x))
        else:
            return 1/(1 + math.exp(-x))

    def tanh(x) :
        return math.tanh(x)

    def linear(x) :
        return x


    methods = {'relu' : relu,'sigmoid' : sigmoid,'tanh' : tanh,'linear' : linear}