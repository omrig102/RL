from specie import Specie
from genome import Genome
from evaluator import Evaluator
from config import Config
import multiprocessing as mp
import random
import sys
import numpy as np
import math
import os
import pickle
from termcolor import colored
import statistics

class Population() :

    def __init__(self) :
        self.species = []
        self.best_genome = None
        self.max_best_fitness = None
        self.generations_not_improving = 0
        self.generations_not_improving_destroy = 0
        self.compatability_threshold = Config.compatability_threshold

    def initPopulation(self,size) :
        self.species = []
        genomes = []
        for _ in range(size) :
            genomes.append(Genome.generateRandomBaseGenome(Config.env.getInputSize(),Config.env.getOutputSize()))
        leader = random.choice(genomes)
        members = genomes
        specie = Specie(leader,members)
        self.species.append(specie)

    @staticmethod
    def savePopulation(generation,population) :
        dir = Config.root_dir + '/models/models-' + str(generation) + '/'
        if(generation % Config.population_save_rate == 0) :
            print('Saving population')
            
            if not os.path.exists(dir):
                os.mkdir(dir)
            with open(dir + 'config.pkl','wb') as fp:
                pickle.dump(Config,fp)
            with open(dir + 'statistics.txt','w') as fp:
                fp.write('best fitness : {}'.format(population.best_genome.fitness))
            with open(dir + 'population.pkl', 'wb') as fp:
                pickle.dump(population, fp)
        if(generation % Config.best_save_rate == 0) :
            if not os.path.exists(dir):
                os.mkdir(dir)
            with open(dir + 'statistics.txt','w') as fp:
                fp.write('best fitness : {}'.format(population.best_genome.fitness))
            with open(dir + 'best.pkl','wb') as fp :
                pickle.dump(population.best_genome,fp)

    @staticmethod
    def loadPopulation(generation) :
        dir = Config.root_dir + '/models/models-' + str(generation) + '/'
        Config.loadConfig(generation,dir)
            
        with open(dir + 'population.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def loadBest(generation) :
        dir = Config.root_dir + '/models/models-' + str(generation) + '/'
        with open(dir + 'best.pkl', 'rb') as f:
            return pickle.load(f)
    def evaluate(self,generation_num) :
        print(colored('**** Running Generation {} ****'.format(generation_num),'green'))
        print(colored('number of species : {}'.format(len(self.species)),'green'))
        self.best_genome = None
        env = Config.env
        genome_index = 0
        if(Config.processes == 1) :
            for specie in self.species :
                members = []
                leader = None
                for member in specie.members :
                    member = Evaluator.evaluate(genome_index,member,env.clone(),Config.steps)
                    members.append(member)
                    if(specie.leader.id == member.id) :
                        leader = member
                    if(self.best_genome is None) :
                        self.best_genome = member
                    elif(self.best_genome.fitness < member.fitness) :
                        self.best_genome = member
                    genome_index += 1
                specie.member = members
                specie.leader = leader
                
        else : 

            members = []
            for specie in self.species :
                members += specie.members
            with mp.Pool(processes=Config.processes) as pool:
                args = []
                for member in members :
                    args.append((genome_index,member,env.clone(),Config.steps))
                    genome_index += 1
                members = pool.starmap(Evaluator.evaluate,args)
                pool.close()
                pool.join()
            specie_members = {}
            for member in members :
                if(self.best_genome is None or self.best_genome.fitness < member.fitness) :
                    self.best_genome = member
                for index,specie in enumerate(self.species) :
                    for specie_member in specie.members :
                        if(member.id == specie_member.id) :
                            if(index not in specie_members) :
                                specie_members[index] = []
                            specie_members[index].append(member)
                            break
            for index,specie in enumerate(self.species) :
                specie.members = specie_members[index]
                for member in specie.members :
                    if(member.id == specie.leader.id) :
                        specie.leader = member


        if(self.max_best_fitness is None or self.max_best_fitness < self.best_genome.fitness) :
            self.max_best_fitness = self.best_genome.fitness
            self.generations_not_improving = 0
            self.generations_not_improving_destroy = 0
            Config.compatability_threshold = self.compatability_threshold
        else :
            self.generations_not_improving += 1
            self.generations_not_improving_destroy += 1
        
        print(colored('best fitness : {}'.format(self.best_genome.fitness),'green'))
        print(colored('best nodes : {}'.format(len(self.best_genome.nodes)),'green'))
        print(colored('best connections : {}'.format(len(self.best_genome.connections)),'green'))



    def roulletteSelection(self,survived_species,sum_survived_adj_fitness) :
        cross_specie_size = 0
        if(np.random.uniform() <= Config.cross_specie_prob) :
            cross_specie_size = random.choice(range(Config.population_size))
        survived_species = sorted(survived_species,key=lambda x : x.average_adj_fitness)
        new_members = []
        species_members_size = []
        for specie in survived_species :
            if(sum_survived_adj_fitness == 0) :
                species_members_size.append(int(math.ceil((Config.population_size - cross_specie_size)/len(survived_species))))
            else :
                species_members_size.append(int(math.ceil((Config.population_size - cross_specie_size)* specie.average_adj_fitness / sum_survived_adj_fitness)))

        sum_species_members_size = sum(species_members_size)
        difference = (Config.population_size - cross_specie_size)/ sum_species_members_size
        species_members_size = [int(round(size * difference)) for size in species_members_size]


        for index,specie in enumerate(survived_species) :
            new_members += specie.generateNewMembers(species_members_size[index])
            specie.members = specie.elites
        for _ in range(cross_specie_size) :
            specie_1 = random.choice(survived_species)
            specie_2 = random.choice(survived_species)
            parent_1 = np.random.choice(specie_1.members,1,specie_1.members_prob)[0]
            parent_2 = np.random.choice(specie_2.members,1,specie_2.members_prob)[0]
            child = Genome.generateNewGenome(parent_1,parent_2)
            new_members.append(child)
        return new_members




    def generateNewPopulation(self,current_generation) : 
        offset = sys.maxsize
        #standard_dev = 0
        #stdev_data = []
        for specie in self.species :
            for member in specie.members :
                #stdev_data.append(member.fitness)
                if(offset > member.fitness) :
                    offset = member.fitness
        
        #standard_dev = statistics.stdev(stdev_data)
        #print('STDEV : {}'.format(standard_dev))
        if(self.generations_not_improving > Config.best_fitness_generations_to_not_improve and Config.compatability_threshold > 0) :
            Config.compatability_threshold -= Config.compatability_threshold_delta

        
        #calculate adjusted fitness
        best_specie = None
        for specie in self.species :
            specie.calculateAdjFitness(offset)
            specie.sortMembersByFitness()
            specie.updateNotImproved()
            if(best_specie is None or best_specie.average_adj_fitness < specie.average_adj_fitness) :
                best_specie = specie
            

        if(self.generations_not_improving_destroy > Config.destroy_population) :
            if(Config.use_species) :
                elites_size = Config.elites_num
                if(len(best_specie.members) < elites_size) :
                    elites_size = len(best_specie.members)
                    
                self.initPopulation(Config.population_size - elites_size)
                best_specie.chooseElites()
                if(best_specie.leader in best_specie.elites) :
                    best_specie.members = best_specie.elites
                else :
                    best_specie.members = [best_specie.leader] + best_specie.elites
                self.species.append(best_specie)
            else :
                self.initPopulation(Config.population_size)
            self.generations_not_improving_destroy = 0
            self.generations_not_improving = 0
            self.generations_to_not_update_mutations = 0
            self.max_best_fitness = - sys.maxsize
            return

        
        survived_species = []
        sum_survived_adj_fitness = 0
        for specie in self.species :
            if(Config.use_species_elites and specie == best_specie) :
                specie.generations_not_improved = 0
                specie.calcMembersProb()
                if(len(self.species) == 1) :
                    specie.eliminateWeakMembers()
                specie.chooseElites()
                survived_species.append(specie)
                sum_survived_adj_fitness += specie.average_adj_fitness
            else :
                specie.calcMembersProb()
                specie.eliminateWeakMembers()
                if(not specie.isStagnant(current_generation) and len(specie.members) > 0) :
                    
                    specie.chooseElites()
                    survived_species.append(specie)
                    sum_survived_adj_fitness += specie.average_adj_fitness

        if(len(survived_species) == 0) :
            self.initPopulation(Config.population_size)

        new_members = self.roulletteSelection(survived_species,sum_survived_adj_fitness)
        for specie in survived_species :
            specie.members = specie.elites

        if(Config.use_species) :
            for member in new_members :
                distances = []
                for specie in survived_species :
                    distances.append(member.distance(specie.leader))
                min_distance = min(distances)
                if(min_distance <= Config.compatability_threshold) :
                    specie_index = np.argmin(distances)
                    survived_species[specie_index].members.append(member)
                else :
                    specie = Specie(leader=member)
                    if(Config.elites_num != 0) :
                        specie.elites = [member]
                    survived_species.append(specie)
                    self.generations_not_improving = 0
                    Config.compatability_threshold = self.compatability_threshold
        else :
            survived_species[0].members += new_members
     

        for specie in survived_species :
            specie.leader = random.choice(specie.elites)
            specie.elites = []

        self.species = survived_species





