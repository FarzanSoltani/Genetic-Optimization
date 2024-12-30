import datetime
import numpy as np
import multiprocessing as mp
from dbconnection import insertParam


last_hour   = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0, second=0, minute=0)
lastts      = int(last_hour.timestamp())

class GeneticStrategy(object):
    def __init__(self,function,num_generations,population_size,mutation_rate,tournament_size,bound,dimension,n_worker,database):
        
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate   = mutation_rate
        self.bound           = bound
        self.dimension       = dimension
        self.tournament_size = tournament_size
        self.pool            = mp.Pool(n_worker)
        self.function        = function
        self.best_resault    = 0
        self.best_variable   = 0
        self.pop_fit         = []
        self.time            = 0
        self.database        = database

    def spawn_population(self,N,size,bound):
        pop = []
        for i in range(N):
            vec = np.random.uniform(low=bound[0], high=bound[1], size=size)
            fit = 0
            p = {'params':vec, 'fitness':fit}
            pop.append(p)
        return pop
        
    def recombine(self, x1 ,x2): 
        x1 = x1['params'] 
        x2 = x2['params']
        l = x1.shape[0]
        split_pt = np.random.randint(l) 
        child1 = np.zeros(l)
        child2 = np.zeros(l)
        child1[0:split_pt] = x1[0:split_pt] 
        child1[split_pt:] = x2[split_pt:]
        child2[0:split_pt] = x2[0:split_pt]
        child2[split_pt:] = x1[split_pt:]
        c1 = {'params':child1, 'fitness': 0.0} 
        c2 = {'params':child2, 'fitness': 0.0}
        return c1, c2
    
    def mutate(self,x, rate,bound):
        x_ = x['params']
        num_to_change = int(rate * x_.shape[0])
        idx  = np.random.choice(range(x_.shape[0]), num_to_change, replace=False)
        rand = np.random.uniform(low=bound[0], high=bound[1], size=self.dimension)
        x_[idx] = rand[idx]
        x['params'] = x_
        return x
    
    def test_model(self,agent):
        score = self.function(agent['params'])
        return score
        
    def next_generation(self,pop,mut_rate,tournament_size):
        new_pop = []
        lp = len(pop)
        while len(new_pop) < len(pop): 
            rids       = np.random.randint(low=0,high=lp,size=(int(tournament_size*lp))) 
            batch      = np.array([[i,x['fitness']] for (i,x) in enumerate(pop) if i in rids]) 
            scores     = batch[batch[:, 1].argsort()] 
            i0, i1     = int(scores[-1][0]),int(scores[-2][0]) 
            parent0,parent1 = pop[i0],pop[i1]
            offspring_ = self.recombine(parent0,parent1) 
            child1     = self.mutate(offspring_[0], rate=mut_rate,bound=self.bound) 
            child2     = self.mutate(offspring_[1], rate=mut_rate,bound=self.bound)
            offspring  = [child1, child2]
            new_pop.extend(offspring)
        return new_pop
        
    def get_best(self,pop):
        array_idx = np.array([p['fitness'] for p in pop]).argmax()
        return pop[array_idx]['params'] , pop[array_idx]['fitness']
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def evaluate_population(self,pop):
        tot_fit = 0
        lp = len(pop)
        scores = self.pool.map(self.test_model, pop)
        for i in range(lp):
            pop[i]['fitness'] = scores[i]
        tot_fit += sum(scores)
        avg_fit = tot_fit / lp
        return pop, avg_fit
    
    def run(self):
        begin = datetime.datetime.now()
        pop = self.spawn_population(N=self.population_size,size=self.dimension,bound=self.bound)
        for i in range(self.num_generations):
            pop, avg_fit = self.evaluate_population(pop)

            best_var ,best_func = self.get_best(pop)

            if best_func > self.best_resault:
                self.best_resault  = best_func
                self.best_variable = best_var
            self.pop_fit.append(avg_fit)
            pop = self.next_generation(pop, mut_rate=self.mutation_rate,tournament_size=self.tournament_size)
            print('Iteration :',i)
        end = datetime.datetime.now()
        self.time = end-begin

        param = dict()
        param["timestamp"]                 = lastts*1000
        param["buylimitpercentage"]        = best_var[0]
        param["selllimitpercentage"]       = best_var[1]
        param["buystoploss"]               = best_var[2]
        param["sellstoploss"]              = best_var[3]
        param["buytakeprofitpercentage"]   = best_var[4]
        param["selltakeprofitpercentage"]  = best_var[5]
        param["trailpercent"]              = best_var[6]

        insertParam(self.database,param)
        
    def running_mean(self,x,n=5):
        conv = np.ones(n)
        y = np.zeros(x.shape[0]-n)
        for i in range(x.shape[0]-n):
            y[i] = (conv @ x[i:i+n]) / n
        return y
