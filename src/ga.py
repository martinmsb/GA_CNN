from cnn import CNN, model_train, model_test
import random

class GeneticAlgorithm:

    def __init__(self, generations, population_size, nparents, tournsize, maxnlayers, c_skernels_upperbound, c_nchannels_upperbound, p_skernels_upperbound,
                probability_crossover, probability_mutation, elitism, train_load, test_load):
        """
        Construct a new Genetic Algorithm
        :param generations: Number of iterations of the algorithm
        :param population_size: Size of the initial population of each generation
        :param nparents: Number of parents selected from each population
        :param tournsize: Number of parents selected for each round of the tournament selection
        :param maxnlayers: Maximum number of layers of each individual (individual size)
        :param c_skernels_upperbound: Maximum kernel size of convolutional layers
        :param c_nchannels_upperbound: Maximum output channels of convolutional layers
        :param p_skernels_upperbound: Maximum kernel size of pooling layers
        :param probability_crossover: Probability of a pair of parents to produce an offspring
        :param probability_mutation: Probability of an offsprint to mutate
        :param elitism: Number of elite individuals. Introduce 0 in order to not use elitism
        :param train_load: Data Loader containing the train data
        :param test_load: Data Loader containing the test data
        :return: -
        """
        self.generations = generations
        self.population_size = population_size
        self.nparents = nparents
        self.tournsize = tournsize
        self.maxnlayers = maxnlayers
        self.c_skernels_upperbound = c_skernels_upperbound
        self.c_nchannels_upperbound = c_nchannels_upperbound
        self.p_skernels_upperbound = p_skernels_upperbound
        # Avoid error if elitism provided is higher than population size
        self.elitism = elitism if elitism < population_size else population_size
        self.elitism_pop = []
        self.probability_crossover = probability_crossover
        self.probability_mutation = probability_mutation
        self.population = []
        self.population_fit = {}
        
        # Change seed manually, add it as parameter or comment this line for introducing more variance in results
        #random.seed(10)

        self.train_load = train_load
        self.test_load = test_load

        # Cache dictionary for storing pairs of individual and fitness value of every generation in order to reduce execution time.
        self.cache_fitness = {}

        # Statistic values
        self.sols_best_inds = []
        self.sols_max_fits = []
        self.sols_avg_fits = []
        
    def population_generation(self):
        """
        Generate the initial population and calculate the fitness of each individual
        :return: -
        """
        for _ in range(self.population_size):
            individual = []
            nlayers = random.randint(2, self.maxnlayers)
            for _ in range(nlayers):
                type = random.uniform(0, 1)
                # If number generated is < 0.5, convolutional layer is added
                if(type < 0.5):
                    skernel = random.randint(1, self.c_skernels_upperbound)
                    nchannel = random.randint(1, self.c_nchannels_upperbound)
                    individual.append(('c', skernel, nchannel))
                # If number generated is in [0.5, 0.75) maxpool layer is added. If is in [0.75, 0] avgpool layer is added.
                else:
                    skernel = random.randint(1, self.p_skernels_upperbound)
                    individual.append(('mp' if type < 0.75 else 'ap', skernel))
            self.population.append(individual)
        self.population_fit = self.population_fitness(self.population)

    def fitness(self, individual):
        """
        Calculate the fitness value (accuracy of the CNN) of an individual
        :param individual: Individual of a population
        return: -
        """
        config = []
        c_skernels = []
        c_nchannels = []
        p_skernels = []
        for layer in individual:
            type = layer[0]
            config.append(type)
            if(type == 'c'):
                c_skernels.append(layer[1])
                c_nchannels.append(layer[2])
            else:
                p_skernels.append(layer[1])

        i_model = CNN(config, c_skernels, c_nchannels, p_skernels)
        model_train(i_model, self.train_load)
        acc = model_test(i_model, self.test_load)
        return acc
    
    def population_fitness(self, population):
        """
        Calculate the fitness value of every individual of a population
        :param population: Population of the generation
        :return: Dictionary containing pairs of individual and fitness value
        """
        fitness = {}
        for individual in population:
            individual_tuple = tuple(individual)
            fit = 0
            if individual_tuple not in self.cache_fitness:
                fit = self.fitness(individual)
                self.cache_fitness[individual_tuple] = fit
            else:
                fit = self.cache_fitness[individual_tuple]
            fitness[individual_tuple] = fit
        return fitness

    def selectParents(self, nparents, tournsize):
        """
        Parents selection using tournament strategy
        :param nparents: Number of parents to select
        :param tournsize: Number of individuals selected for each tournament iteration
        :return: -
        """
        self.parents = []

        for _ in range(nparents):
            candidates_fit = {}
            candidates = [random.choice(self.population) for _ in range(tournsize)]
            for candidate in candidates:
                fit = 0    
                candidate_tuple = tuple(candidate)
                fit = self.population_fit[candidate_tuple]
                candidates_fit[candidate_tuple] = fit
            parent = max(candidates_fit, key = candidates_fit.get)
            self.parents.append(list(parent))
        
        print(self.parents)
    
    def crossover(self):
        """
        Selects pairs of perentes and perform the crossover with a specific probability generating the new population of offsprings until the population size is reached
        :return: -
        """
        self.new_pop = []
        while len(self.new_pop) < self.population_size:
            prob = random.uniform(0, 1)
            p1 = random.choice(self.parents)
            p2 = random.choice(self.parents)
            # If parents are the same individual reselect one of them
            while p2 == p1:
                p2 = random.choice(self.parents)
            # Check is crossover is performed
            if(prob < self.probability_crossover):
                # Crossover is peformed and childs are added to new generation
                div = random.randint(1, min(len(p1), len(p2))-1)
                c1 = p1[:div] + p2[div:]
                c2 = p2[:div] + p1[div:]
            else:
                # Crossover is not performed and parents are added to new generation
                c1 = p1
                c2 = p2
            self.new_pop.append(c1)
            self.new_pop.append(c2)
    
    def mutation(self):
        """
        Using the new population of offsprings generated, perform the mutation process over them 
        :return: -
        """
        mutation_pop = []
        for individual in self.new_pop:
            prob = random.uniform(0, 1)
            # Check if mutation is performed
            if(prob < self.probability_mutation):
                position = random.randint(0, len(individual)-1)
                mutation = individual
                type = random.uniform(0, 1)
                # If number generated is < 0.5, convolutional layer is added
                if(type < 0.5):
                    skernel = random.randint(1, self.c_skernels_upperbound)
                    nchannel = random.randint(1, self.c_nchannels_upperbound)
                    mutation[position] = ('c', skernel, nchannel)
                # If number generated is in [0.5, 0.75) maxpool layer is added. If is in [0.75, 0] avgpool layer is added.
                else:
                    skernel = random.randint(1, self.p_skernels_upperbound)
                    mutation[position] = ('mp' if type < 0.75 else 'ap', skernel)
                mutation_pop.append(mutation)
            else:
                mutation_pop.append(individual)
        self.new_pop = mutation_pop
    
    def evolve(self):
        """
        Calculate the fitness of the new population of offsprings, add elite individuals if necessary and store results as population for next generation
        :return: -
        """
        new_pop_fit = self.population_fitness(self.new_pop)
        # Check if elitism activated
        if self.elitism > 0:
            # Remove from offsprings as many worst elements as number of elitism individuals
            individuals_to_delete = sorted(new_pop_fit, key=lambda d: d[1])[:self.elitism]
            for ind in individuals_to_delete:
                self.new_pop.remove(list(ind))
            # Add elite individuals to offsprings
            for elite in self.elitism_pop:
                self.new_pop.append(list(elite))
        # Calculate the fitness of the new population and store them as population for next generation
        self.population_fit = self.population_fitness(self.new_pop)
        self.population = self.new_pop
    
    def store_population_data(self):
        """
        Store at the end of each generation process the best individual, its fitness value and the average fitness value of the population
        :return: A tuple containing best individual, its fitness value and the average fitness value of the population
        """
        best_ind = max(self.population_fit, key=self.population_fit.get)
        best_fit = self.population_fit.get(best_ind)
        list_pop_fit = list(self.population_fit.values())
        total_fit = 0
        for individual in self.population:
            individual_tuple = tuple(individual)
            individual_fit = self.cache_fitness.get(individual_tuple)
            total_fit+=individual_fit
        avg_fit = total_fit / self.population_size
        self.sols_best_inds.append(list(best_ind))
        self.sols_max_fits.append(best_fit)
        self.sols_avg_fits.append(avg_fit)

        return (best_ind, best_fit, avg_fit)

    def exec(self):
        """
        Execute for each generation the Genetic Algorithm phases
        :return: -
        """
        print("Generating initial population...")
        self.population_generation()
        for generation in range(self.generations):
            print("Generation " + str(generation) + " population:")
            for pop in self.population:
                print(pop)
            # If elitism is declared collect the elite individuals
            if self.elitism > 0:
                self.elitism_pop = sorted(self.population_fit, key=self.population_fit.get, reverse=True)[:self.elitism]
            print("Selecting parents...")
            self.selectParents(self.nparents, self.tournsize)
            conv = all(x == self.parents[0] for x in self.parents)
            if(conv):
                print("The solutions converged. Ending the execution...")
                return
            self.crossover()
            print("Crossover...")
            
            self.mutation()
            print("Mutation...")
            
            self.evolve()
            print("Evolving...")
            
            results = self.store_population_data()
            print("Results of the generation: ")
            print("Best individual: ")
            print(results[0])
            print("Maximum fitness: " + str(results[1]) + ". Average fitness: " + str(results[2]) + ".")