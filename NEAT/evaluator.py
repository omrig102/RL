from environment import Environment
from config import Config

class Evaluator() :

    @staticmethod
    def evaluate(generation_num,genome,env,steps,render=False) :
        
        genome.create_model()
        env.initialize()

        state = env.preprocess(env.reset())

        total_fitness = 0
        for step in range(steps) :
            if(render) :
                env.render()
            action = env.actionProcessor(genome.run(state))
            next_state,fitness,done,info = env.step(action)

            total_fitness += fitness
            state = env.preprocess(next_state)
            if(done) :
                env.close()
                genome.fitness = total_fitness
                print('Genome {} , fitness : {}'.format(generation_num,total_fitness))
                return genome


        env.close()
        genome.fitness = total_fitness
        print('Genome {} , fitness : {}'.format(generation_num,total_fitness))
        return genome