from config import Config
import math
import random
from genome import Genome
import numpy as np
import sys

class Specie() :

    def __init__(self,leader=None,members=None) :
        self.leader = leader
        if(members is None) :
            self.members = []
            if(self.leader is not None) :
                self.members.append(self.leader)
        else :
            self.members = members
        self.average_adj_fitness = 0
        self.sum_adj_fitness = None
        self.prev_historical_average_adj_fitness = []
        self.generations_not_improved = 0
        self.elites = []
        self.members_prob = []
    


    def chooseMembers(self) :
        if(len(self.members) == 1) :
            return self.members[0]
        if(self.sum_adj_fitness == 0) :
            return random.choice(self.members)
        zero_fitness_members = [member for member in self.members if(member.fitness == 0)]
        if(len(zero_fitness_members) == len(self.members)) :
            return random.choice(self.members)
        while(True) :
            for member in self.members :
                if(np.random.uniform() <= member.fitness / self.sum_adj_fitness) :
                    return member
 
    def generateNewMembers(self,members_size) :
        new_members = []
        for _ in range(members_size) :
            parents = np.random.choice(self.members,2,self.members_prob)
            new_members.append(Genome.generateNewGenome(parents[0],parents[1]))
        
        return new_members

    def isStagnant(self,current_generation) :
        if(self.generations_not_improved > Config.generations_to_not_improve) :
            return True
        return False

    def updateNotImproved(self) :
        improved = False
        if(not self.prev_historical_average_adj_fitness) :
            self.prev_historical_average_adj_fitness.append(self.max_fitness)
            return
        max_prev_fitness = max(self.prev_historical_average_adj_fitness)
        if(max_prev_fitness < self.max_fitness) :
            self.generations_not_improved = 0
        else :
            self.generations_not_improved += 1
        
        if(len(self.prev_historical_average_adj_fitness) >= 10) :
            self.prev_historical_average_adj_fitness.pop(0)
        
        self.prev_historical_average_adj_fitness.append(self.max_fitness)

    def sortMembersByFitness(self) :
        self.members = sorted(self.members,key=lambda x : x.fitness)

    def chooseElites(self) :
        if(Config.elites_num != 0) :
            self.elites = self.members[-Config.elites_num : ]

    def calcMembersProb(self) :
        self.members_prob = []
        for member in self.members :
            self.members_prob.append(member.fitness/self.sum_adj_fitness)

    def eliminateWeakMembers(self) :
        start = int(math.ceil(len(self.members) * Config.elimination))
        if(len(self.members) - start < Config.elites_num) :
            return
        if(start >= len(self.members)) :
            self.members = []
            return
        self.members = self.members[start : ]
        if(self.leader not in self.members) :
            self.leader = random.choice(self.members)
        self.calcMembersProb()


    def calculateAdjFitness(self,offset) :
        noise = 0.07
        self.average_adj_fitness = 0
        self.max_fitness = - sys.maxsize
        for member in self.members :
            fitness = member.fitness
            member.fitness += abs(offset) + noise
            member.fitness /= len(self.members)
            if(self.max_fitness < fitness) :
                self.max_fitness = fitness
            self.average_adj_fitness += member.fitness

        self.sum_adj_fitness = self.average_adj_fitness
        self.average_adj_fitness /= len(self.members)


