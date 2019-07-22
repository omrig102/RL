import random
import numpy as np
from config import Config
from node import Node
from connection import Connection
from activation import Activation

class Genome() :

    def __init__(self,nodes,connections,id) :
        self.nodes = nodes
        self.connections = connections
        self.fitness = 0
        self.id = id
        self.__divideNodes()

    def __divideNodes(self) :
        self.input_nodes = {}
        self.hidden_nodes = {}
        self.output_nodes = {}
        self.input_connections = {}

        for id,node in self.nodes.items() :
            if(node.type == 'input') :
                self.input_nodes[id] = node
            elif(node.type == 'hidden') :
                self.hidden_nodes[id] = node
            else :
                self.output_nodes[id] = node

        self.input_connections = Genome.__getNodesInputConnections(self.connections)

    @staticmethod
    def __getNodesInputConnections(connections) :
        input_connections = {}
        for _,connection in connections.items() :
            output = connection.output
            if(output.id not in input_connections) :
                input_connections[output.id] = []
            input_connections[output.id].append(connection)
        return input_connections

    def create_model(self) :
        self.diversity_reward = 0
        self.actions = []
        return self

    def run(self,inputs) :
        for _,node in self.nodes.items() :
            node.value = None

        outputs = [0] * len(self.output_nodes)
        for id,node in self.output_nodes.items() :
            node.value = self.__calculateNodeValue(node,inputs)
            if(node.type == 'output') :
                outputs[node.order] = node.value

        return outputs

    def __calculateNodeValue(self,node,inputs) :
        if(node.type == 'input') :
            node.value = inputs[node.order]
            return node.activation(node.value)
        if(node.id not in self.input_connections) :
            node.value = 0
            return node.value
        connections = self.input_connections[node.id]
        value = 0
        for connection in connections : 
            if(connection.enabled) :
                input_node = self.nodes[connection.input.id]
                if(input_node.value is None) :
                    input_node.value = self.__calculateNodeValue(input_node,inputs)
                value += input_node.value * connection.weight
        
        return node.activation(value)              

    def distance(self,genome) :
        if(not self.connections) :
            if(genome.connections) :
                return Config.c1 * len(genome.connections) + Config.c2 * 0 + Config.c3 * 0
            return 0
        else :
            if(not genome.connections) :
                return Config.c1 * 0 + Config.c2 * len(self.connections) + Config.c3 * 0
                
        
        all_connections = [connection for _,connection in self.connections.items()]
        all_connections += [connection for _,connection in genome.connections.items()]

        N = max(len(self.connections),len(genome.connections))
        min_innovation = min(max(self.connections),max(genome.connections))

        matching = 0
        average_weights = 0
        excess = 0
        disjoint = 0

        for innovation,connection in self.connections.items() :
            if(innovation in genome.connections) :
                matching += 1
                average_weights += abs(connection.weight - genome.connections[innovation].weight)
            elif(innovation < min_innovation) :
                disjoint += 1
            else :
                excess += 1

        for innovation,connection in genome.connections.items() :
            if(innovation in self.connections) :
                matching += 1
                average_weights += abs(connection.weight - self.connections[innovation].weight)
            elif(innovation < min_innovation) :
                disjoint += 1
            else :
                excess += 1
        
        if(matching != 0) :
            average_weights /= matching

        return (Config.c1 * disjoint) / N + (Config.c2 * excess) / N + Config.c3 * average_weights
        


    @staticmethod
    def generateRandomBaseGenome(input_size,output_size) :
        input_nodes = {}
        for i in range(input_size) :
            id = Config.getNodeId()
            activation = random.choice(list(Activation.methods.items()))[1]
            node = Node('input',activation,id)
            node.order = i
            input_nodes[id] = node
        
        output_nodes = {}
        for i in range(output_size) :
            id = Config.getNodeId()
            activation = random.choice(list(Activation.methods.items()))[1]
            node = Node('output',activation,id)
            node.order = i
            output_nodes[id] = node

        connections = {}
        for input_id,input_node in input_nodes.items() :
            for output_id,output_node in output_nodes.items() :
                if(np.random.uniform() <= 0.5) :
                    weight = random.uniform(-1,1)
                    innovation = Config.getInnovationId()
                    if(np.random.uniform() <= 0.5) :
                        enabled = True
                    else :
                        enabled = False
                    connection = Connection(input_node,output_node,innovation,weight,enabled)
                    connections[innovation] = connection
        
        nodes = {}
        for id,node in input_nodes.items() :
            nodes[id] = node
        for id,node in output_nodes.items() :
            nodes[id] = node
        genome = Genome(nodes,connections,Config.getGenomeId())
        return genome

    @staticmethod
    def generateNewGenome(parent_1,parent_2) :
        if(parent_1.fitness > parent_2.fitness) :
            fittest_parent = parent_1
            other_parent = parent_2
        elif(parent_1.fitness < parent_2.fitness) :
            fittest_parent = parent_2
            other_parent = parent_1
        elif(np.random.uniform() <= 0.5) :
            fittest_parent = parent_1
            other_parent = parent_2
        else :
            fittest_parent = parent_2
            other_parent = parent_1


        fittest_connections = fittest_parent.connections
        other_connections = other_parent.connections

        child_nodes = {}
        child_connections = {}

        for key,fittest_connection in fittest_connections.items() :
            chosen_connection = None
            if(key in other_connections) :
                Genome.__handle_matching_connetions(fittest_connection,other_connections[key],child_nodes,child_connections)

            else :
                Genome.__handle_disjoint_or_excess(fittest_connection,child_nodes,child_connections)

        for id,node in fittest_parent.nodes.items() :
            if(id not in child_nodes) :
                child_nodes[id] = node.clone()

        Genome.__mutate(child_nodes,child_connections)

        child = Genome(child_nodes,child_connections,Config.getGenomeId())

        return child


    @staticmethod
    def __handle_matching_connetions(fittest_connection,other_connection,child_nodes,child_connections) :
        if(np.random.uniform() <= 0.5) :
            child_connection = fittest_connection.clone()
            child_nodes[child_connection.input.id] = child_connection.input
            child_nodes[child_connection.output.id] = child_connection.output
        else :
            input_node = fittest_connection.input.clone()
            input_node.activation = other_connection.input.activation
            output_node = fittest_connection.output.clone()
            output_node.activation = other_connection.output.activation
            child_connection = Connection(input_node,output_node,other_connection.innovation,other_connection.weight,other_connection.enabled)
            child_nodes[input_node.id] = input_node
            child_nodes[output_node.id] = output_node

        child_connections[child_connection.innovation] = child_connection

    @staticmethod
    def __handle_disjoint_or_excess(fittest_connection,child_nodes,child_connections) :
        child_connection = fittest_connection.clone()
        child_nodes[child_connection.input.id] = child_connection.input
        child_nodes[child_connection.output.id] = child_connection.output
        child_connections[child_connection.innovation] = child_connection

    @staticmethod
    def __mutate(nodes,connections) :
        Genome.__mutation_weight(connections)
        Genome.__mutation_enabled(connections)
        Genome.__mutation_remove_connection(connections)
        Genome.__mutation_add_connection(nodes,connections)
        Genome.__mutation_remove_node(nodes,connections)
        Genome.__mutation_add_node(nodes,connections)

    @staticmethod
    def __mutation_activation(nodes) :
        if(np.random.uniform() <= Config.mutation_activation_rate) :
            node = random.choice(list(nodes.items()))[1]
            node.activation = random.choice(list(Activation.methods.items()))[1]

    @staticmethod
    def __mutation_weight(connections) :
        if(not connections) :
            return         
        if(np.random.uniform() <= Config.mutation_weight_rate) :
            connection = random.choice(list(connections.items()))[1]
            connection.weight += random.uniform(Config.mutation_min,Config.mutation_max)

    @staticmethod
    def __mutation_enabled(connections) :
        if(not connections) :
            return 
        if(np.random.uniform() <= Config.mutation_enabled_rate) :
            connection = random.choice(list(connections.items()))[1]
            if(connection.enabled) :
                connection.enabled = False
            else :
                connection.enabled = True
    
    @staticmethod
    def __mutation_add_connection(nodes,connections) :
        if(np.random.uniform() <= Config.mutation_new_connection_rate) :
            node_1 = nodes[random.choice(list(nodes.keys()))]
            
            candidates = {}
            for id,node in nodes.items() :
                if(node_1.type == node.type and node.type != 'hidden') :
                    continue
                if(node_1.id == node.id) :
                    continue
                candidates[id] = node
                
            for _,connection in connections.items() :
                input_node = connection.input
                output_node = connection.output
                if(node_1.id == input_node.id) :
                    node = output_node
                elif(node_1.id == output_node.id) :
                    node = input_node
                
                if(node.id in candidates) :
                    del candidates[node.id]

            if(candidates) :
                node_2 = nodes[random.choice(list(candidates.keys()))]
                innovation = Config.getInnovationId()
                if(node_1.type == 'input') :
                    new_connection = Connection(node_1,node_2,innovation,1,True)
                elif(node_1.type == 'output') :
                    new_connection = Connection(node_2,node_1,innovation,1,True)
                else :
                    new_connection = Connection(node_1,node_2,innovation,1,True)

                connections[innovation] = new_connection

                Genome.__fix_cycles(nodes,connections)

    @staticmethod
    def __fix_cycles(child_nodes,child_connections) :
        input_connections = Genome.__getNodesInputConnections(child_connections)
        for id,node in child_nodes.items() :
            connections_to_remove = Genome.__fix_cycles_helper(node,None,input_connections,{})
            for connection in connections_to_remove : 
                if(connection.innovation in child_connections) :
                    del child_connections[connection.innovation]

    @staticmethod
    def __fix_cycles_helper(node,connection,input_connections,visited) :
        if(node.type == 'input' or node.id not in input_connections) :
            return []
        if(node.id in visited) :
            return [connection]
        visited[node.id] = node
        node_connections = input_connections[node.id]
        connections_to_remove = []
        for input_connection in node_connections :
            connections_to_remove += Genome.__fix_cycles_helper(input_connection.input,input_connection,input_connections,visited)

        return connections_to_remove


    @staticmethod
    def __mutation_remove_connection(connections) :
        if(not connections) :
            return 
        if(np.random.uniform() <= Config.mutation_remove_connection_rate) :
            connection = random.choice(list(connections.items()))[1]
            del connections[connection.innovation]

    @staticmethod
    def __mutation_add_node(nodes,connections) :
        if(not connections) :
            return 
        if(np.random.uniform() <= Config.mutation_new_node_rate) :
            connection = random.choice(list(connections.items()))[1]
            connection.enabled = False
            new_node = Node('hidden',connection.input.activation,Config.getNodeId())
            input_1 = connection.input
            new_connection_1 = Connection(input_1,new_node,innovation=Config.getInnovationId(),weight=1,enabled=True)
            output_2 = connection.output
            new_connection_2 = Connection(new_node,output_2,innovation=Config.getInnovationId(),weight=connection.weight,enabled=True)
            nodes[new_node.id] = new_node
            connections[new_connection_1.innovation] = new_connection_1
            connections[new_connection_2.innovation] = new_connection_2

    @staticmethod
    def __mutation_remove_node(nodes,connections) :
        if(np.random.uniform() <= Config.mutation_remove_node_rate) :
            hidden_nodes = [node.id for _,node in nodes.items() if node.type == 'hidden']
            if(len(hidden_nodes) != 0) :
                node_id = random.choice(hidden_nodes)
                del nodes[node_id]
                connections_to_remove = []
                for index,connection in connections.items() :
                    if(connection.input.id == node_id or connection.output.id == node_id) :
                        connections_to_remove.append(index)

                for connection in connections_to_remove :
                    del connections[connection]