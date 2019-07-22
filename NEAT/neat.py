from config import Config
from evaluator import Evaluator
from population import Population

class Neat() :

    def __init__(self) :
        self.population = Population()
        

    def run(self) :
        best_fitness = 0
        current_generation = Config.start_generation
        if(Config.start_generation == 0) :
            self.population.initPopulation(Config.population_size)
        else :
            self.population = Population.loadPopulation(Config.start_generation)
        while(best_fitness < Config.max_fitness and current_generation < Config.max_generations) :
            self.population.evaluate(current_generation)
            best_genome = self.population.best_genome
            best_fitness = best_genome.fitness
            Population.savePopulation(current_generation,self.population)
            self.population.generateNewPopulation(current_generation)
            current_generation += 1