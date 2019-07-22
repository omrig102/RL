from gym_environment import GymEnvironment
from xor_environment import XOREnvironment
from mario_environment import MarioEnvironment
from retro_environment import RetroEnvironment
import pickle

class Config() :

    population_size = 25
    steps = 7000
    max_fitness = 300
    max_generations = 1000000
    processes = 4
    start_generation = 0
    genome_id = 0
    node_id = 0
    innovation_id = 0
    render = True
    root_dir = '.'
    population_save_rate = 5 
    best_save_rate = 1
    mutation_activation_rate = 0.2
    mutation_weight_rate = 0.5
    mutation_enabled_rate = 0.5
    mutation_new_node_rate = 0.3
    mutation_remove_node_rate = 0.3
    mutation_remove_connection_rate = 0.5
    mutation_new_connection_rate = 0.5
    mutation_min = -2
    mutation_max = 2
    cross_specie_prob = 0.1
    destroy_population = 100
    use_species = True
    c1 = 1
    c2 = 1
    c3 = 0.4
    compatability_threshold = 2
    compatability_threshold_delta = 0.1
    elimination = 0.2
    elites_num = 4
    use_species_elites = True
    generations_to_not_improve = 15
    best_fitness_generations_to_not_improve = 5
    #env = RetroEnvironment('MortalKombatII-Genesis',13,13,use_pixels=True)
    env = GymEnvironment('BipedalWalker-v2',13,13,use_pixels=False)

    @classmethod
    def loadConfig(cls,generation,dir) :
        with open(dir + 'config.pkl','rb') as f:
            config = pickle.load(f)
            cls.population_size = config.population_size
            cls.steps = config.steps
            cls.max_fitness = config.max_fitness
            cls.max_generations = config.max_generations
            cls.processes = config.processes
            cls.start_generation = generation
            cls.genome_id = config.genome_id
            cls.node_id = config.node_id
            cls.innovation_id = config.innovation_id
            cls.render = config.render
            cls.root_dir = config.root_dir
            cls.population_save_rate = config.population_save_rate
            cls.best_save_rate = config.best_save_rate
            cls.mutation_activation_rate = config.mutation_activation_rate
            cls.mutation_weight_rate = config.mutation_weight_rate
            cls.mutation_enabled_rate = config.mutation_enabled_rate
            cls.mutation_new_node_rate = config.mutation_new_node_rate
            cls.mutation_remove_node_rate = config.mutation_remove_node_rate
            cls.mutation_remove_connection_rate = config.mutation_remove_connection_rate
            cls.mutation_new_connection_rate = config.mutation_new_connection_rate
            cls.mutation_min = config.mutation_min
            cls.mutation_max = config.mutation_max
            cls.cross_specie_prob = config.cross_specie_prob
            cls.destroy_population = config.destroy_population
            cls.use_species = config.use_species
            cls.c1 = config.c1
            cls.c2 = config.c2
            cls.c3 = config.c3
            cls.compatability_threshold = config.compatability_threshold
            cls.compatability_threshold_delta = config.compatability_threshold_delta
            cls.elimination = config.elimination
            cls.elites_num = config.elites_num
            cls.use_species_elites = config.use_species_elites
            cls.generations_to_not_improve = config.generations_to_not_improve
            cls.best_fitness_generations_to_not_improve = config.best_fitness_generations_to_not_improve
            cls.env = config.env
    @classmethod
    def getGenomeId(cls) :
        id = cls.genome_id
        cls.genome_id += 1
        return id

    @classmethod
    def getNodeId(cls) :
        id = cls.node_id
        cls.node_id += 1
        return id

    @classmethod
    def getInnovationId(cls) :
        id = cls.innovation_id
        cls.innovation_id += 1
        return id