from evaluator import Evaluator
from environment import Environment
from gym_environment import GymEnvironment
from xor_environment import XOREnvironment
from mario_environment import MarioEnvironment
from retro_environment import RetroEnvironment
from genome import Genome
from population import Population
import sys

def simulate(env,generation) :
    genome = Population.loadBest(generation)
    genome = Evaluator.evaluate(generation,genome,env,1000000,render=True)
    print('Game Ended : {}'.format(genome.fitness))

def simulateXOR(env,generation) :
	genome = Population.loadBest(generation)
	env.initialize()
	frame = env.preprocess(env.reset())
	for i in range(4) :

		action = env.actionProcessor(genome.run(frame))
		
		print('{} : {}'.format(frame,action))
		new_frame,reward,done,info = env.step(action)
		frame = env.preprocess(new_frame)
	env.close()

generation = int(sys.argv[1])
env = GymEnvironment('BipedalWalker-v2',13,13,use_pixels=False)
#env = RetroEnvironment('MortalKombatII-Genesis',13,13,use_pixels=True)
#env = XOREnvironment()
simulate(env,generation)