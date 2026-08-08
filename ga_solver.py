import numpy as np

class EVStationGeneticSolver:
    """
    ماژول الگوریتم ژنتیک برای مکان‌یابی و ظرفیت‌سنجی ایستگاه‌های شارژ خودروهای برقی
    بر اساس ترافیک معابر و هزینه احداث.
    """
    def __init__(self, num_candidate_nodes=10, pop_size=50, generations=100, mutation_rate=0.08):
        self.num_nodes = num_candidate_nodes
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
    def generate_initial_population(self):
        # کروموزوم: آرایه‌ای با طول num_nodes (مقادیر 0 تا 3 نشان‌دهنده تعداد پورت‌های شارژ)
        return np.random.randint(0, 4, size=(self.pop_size, self.num_nodes))

    def calculate_fitness(self, population, traffic_data, cost_per_charger=15.0, revenue_per_traffic=2.5):
        """
        ارزیابی برازندگی: بیشینه‌سازی پوشش ترافیک منفی هزینه‌های احداث
        """
        fitness_scores = []
        for chrom in population:
            total_chargers = np.sum(chrom)
            # محاسبه پوشش ترافیک (حد اشباع خدمت‌رسانی)
            served_traffic = np.minimum(chrom * 25, traffic_data) 
            
            revenue = np.sum(served_traffic) * revenue_per_traffic
            cost = total_chargers * cost_per_charger
            
            # جریمه عدم احداث حداقل یک ایستگاه
            penalty = 0 if total_chargers > 0 else 1000 
            
            fitness = revenue - cost - penalty
            fitness_scores.append(fitness)
            
        return np.array(fitness_scores)

    def selection(self, population, fitness_scores):
        # انتخاب تورنمنتی
        selected = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(self.pop_size, 2, replace=False)
            selected.append(population[i] if fitness_scores[i] > fitness_scores[j] else population[j])
        return np.array(selected)

    def crossover(self, parent1, parent2):
        if np.random.rand() < 0.8:
            point = np.random.randint(1, self.num_nodes)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, chromosome):
        for i in range(self.num_nodes):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = np.random.randint(0, 4)
        return chromosome

    def solve(self, traffic_data):
        population = self.generate_initial_population()
        history = []
        
        best_solution = None
        best_fitness = -np.inf
        
        for gen in range(self.generations):
            fitness_scores = self.calculate_fitness(population, traffic_data)
            max_idx = np.argmax(fitness_scores)
            
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_solution = population[max_idx].copy()
                
            history.append(best_fitness)
            
            # الگوریتم تکاملی
            selected_pop = self.selection(population, fitness_scores)
            next_pop = []
            
            for i in range(0, self.pop_size, 2):
                p1, p2 = selected_pop[i], selected_pop[(i + 1) % self.pop_size]
                c1, c2 = self.crossover(p1, p2)
                next_pop.append(self.mutate(c1))
                next_pop.append(self.mutate(c2))
                
            population = np.array(next_pop)
            
        return best_solution, best_fitness, history