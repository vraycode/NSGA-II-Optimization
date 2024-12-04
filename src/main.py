import random

# 1. initialize population
def initialize_population(pop_size, num_cities, num_items):
    """
    Randomly initialize the population
    pop_size: Population size
    num_cities: Number of cities
    num_items: Number of items
    """
    population = []
    for _ in range(pop_size):
        # Randomly generate path (order of cities)
        path = random.sample(range(num_cities), num_cities)
        # Randomly select items in the backpack (0 means don't select, 1 means select)
        items = [random.choice([0, 1]) for _ in range(num_items)]
        population.append((path, items))  # Each solution contains a path and item selection
    return population


# 2. calculate fitness
def calculate_fitness(path, items, distance_matrix, item_values):
    """
    Calculate the fitness of an individual
    path: City path
    items: Item selection
    distance_matrix: Distance matrix between cities
    item_values: List of item values
    """
    total_distance = 0
    total_value = 0
    # Calculate the path length
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i]][path[i + 1]]
    # Calculate the distance from the last city to the first city
    total_distance += distance_matrix[path[-1]][path[0]]

    # Calculate item value
    total_value = sum([item_values[i] for i in range(len(items)) if items[i] == 1])

    return total_distance, total_value


# 3. nondominated sort
def non_dominated_sort(population, distance_matrix, item_values):
    """
    Sort the population based on non-domination
    """
    fronts = []  # Used to store solutions in different fronts
    # Compare each pair of individuals and check the domination relationship
    dominance = {i: [] for i in range(len(population))}

    # Compare each pair of individuals and check the domination relationship
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            fitness_i = calculate_fitness(population[i][0], population[i][1], distance_matrix, item_values)
            fitness_j = calculate_fitness(population[j][0], population[j][1], distance_matrix, item_values)

            # Check if i dominates j
            if fitness_i[0] <= fitness_j[0] and fitness_i[1] >= fitness_j[1]:
                dominance[i].append(j)
            # Check if j dominates i
            elif fitness_j[0] <= fitness_i[0] and fitness_j[1] >= fitness_i[1]:
                dominance[j].append(i)

    return fronts


# 4. tournament selection
def tournament_selection(population, distance_matrix, item_values):
    """
    Tournament selection operation to select parent individuals
    """
    tournament_size = 2  # Select two individuals for the tournament
    selected_parents = []
    for _ in range(len(population) // 2):  
        contenders = random.sample(population, tournament_size)
        parent = min(contenders, key=lambda x: calculate_fitness(x[0], x[1], distance_matrix, item_values)[0])  # Select the one with the shortest path
        selected_parents.append(parent)
    return selected_parents


# 5. crossover
def crossover(parent1, parent2):
    """
  Single-point crossover operation
    """
    point = random.randint(1, len(parent1[0]) - 1)  # Randomly choose a crossover point
    child1 = (parent1[0][:point] + parent2[0][point:], parent1[1][:point] + parent2[1][point:])
    child2 = (parent2[0][:point] + parent1[0][point:], parent2[1][:point] + parent1[1][point:])
    return child1, child2


# 6. mutate
def mutate(individual):
    """
    Mutation operation: Swap two cities in the path
    """
    path, items = individual
    i, j = random.sample(range(len(path)), 2)
    path[i], path[j] = path[j], path[i]  # 交换路径中的两个城市
    return path, items


# 7. merge & sort
def merge_and_sort(parents, offspring, distance_matrix, item_values):
    """
    Merge parents and offspring, perform non-dominated sorting and select the next generation
    """
    combined_population = parents + offspring
    fronts = non_dominated_sort(combined_population, distance_matrix, item_values)
    return fronts[0]  # Return the first front


# 8. main
def nsga2_algorithm(num_generations, pop_size, num_cities, num_items, distance_matrix, item_values):
    """
    Execute the NSGA-II algorithm
    """
    population = initialize_population(pop_size, num_cities, num_items)
    
    for generation in range(num_generations):
        parents = tournament_selection(population, distance_matrix, item_values)
        offspring = []
        
        # Crossover operation
        for i in range(0, len(parents), 2):
            child1, child2 = crossover(parents[i], parents[i + 1])
            offspring.append(child1)
            offspring.append(child2)
        
        # Mutation operation
        offspring = [mutate(child) for child in offspring]
        
        # Merge parents and offspring
        population = merge_and_sort(parents, offspring, distance_matrix, item_values)

    return population


