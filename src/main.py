import random

# 1. Initialize Population
def initialize_population(pop_size, num_cities, num_items):
    population = []
    for _ in range(pop_size):
        path = random.sample(range(num_cities), num_cities)  # Random path
        items = [random.choice([0, 1]) for _ in range(num_items)]  # Random items selection
        population.append((path, items))
    return population


# 2. Calculate Fitness
def calculate_fitness(path, items, distance_matrix, item_values):
    total_distance = 0
    total_value = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i]][path[i + 1]]
    total_distance += distance_matrix[path[-1]][path[0]]  # Return to the start
    total_value = sum([item_values[i] for i in range(len(items)) if items[i] == 1])
    return total_distance, total_value  # Return both distance and value


# 3. Tournament Selection
def tournament_selection(population, distance_matrix, item_values):
    tournament_size = 2  # Select two individuals for the tournament
    selected_parents = []
    for _ in range(len(population) // 2):  
        contenders = random.sample(population, tournament_size)
        parent = min(contenders, key=lambda x: calculate_fitness(x[0], x[1], distance_matrix, item_values)[0])  # Select the one with the shortest path
        selected_parents.append(parent)
    return selected_parents


# 4. Crossover (Simple single-point crossover)
def crossover(parents):
    offspring = []
    for i in range(0, len(parents) - 1, 2):  # Skip the last one if odd number of parents
        point = random.randint(1, len(parents[i][0]) - 1)  # Random crossover point
        child1 = (parents[i][0][:point] + parents[i+1][0][point:], parents[i][1][:point] + parents[i+1][1][point:])
        child2 = (parents[i+1][0][:point] + parents[i][0][point:], parents[i+1][1][:point] + parents[i][1][point:])
        offspring.append(child1)
        offspring.append(child2)
    if len(parents) % 2 == 1:
        offspring.append(parents[-1])
    return offspring


# 5. Mutation (Simple swap mutation)
def mutate(individual):
    path, items = individual
    i, j = random.sample(range(len(path)), 2)
    path[i], path[j] = path[j], path[i]  # Swap two cities in the path
    return path, items


# 6. Simple Selection (Keep the best individuals)
def select_best(population, num_parents, distance_matrix, item_values):
    sorted_population = sorted(population, key=lambda x: calculate_fitness(x[0], x[1], distance_matrix, item_values)[0])
    return sorted_population[:num_parents]  # Select the best individuals


# 7. Main Function - Simplified NSGA-II Algorithm
def nsga2_algorithm(num_generations, pop_size, num_cities, num_items, distance_matrix, item_values):
    population = initialize_population(pop_size, num_cities, num_items)
    for generation in range(num_generations):
        parents = tournament_selection(population, distance_matrix, item_values)
        offspring = crossover(parents)
        offspring = [mutate(child) for child in offspring]

        # Combine parents and offspring, and select the best individuals
        combined_population = parents + offspring
        population = select_best(combined_population, pop_size, distance_matrix, item_values)

        print(f"Generation {generation + 1}:")
        for individual in population:
            path, items = individual
            path_fitness, item_fitness = calculate_fitness(path, items, distance_matrix, item_values)
            print(f"Path: {path}, Items: {items}, Path Fitness: {path_fitness}, Item Fitness: {item_fitness}")
    return population


# 8. Test the Algorithm
num_cities = 5
num_items = 3
pop_size = 10
num_generations = 5

# Example distance matrix (cities distances)
distance_matrix = [
    [0, 2, 9, 10, 4],
    [2, 0, 6, 7, 3],
    [9, 6, 0, 4, 8],
    [10, 7, 4, 0, 5],
    [4, 3, 8, 5, 0]
]

# Example item values (values for each item in the backpack)
item_values = [10, 20, 30]

# Run the simplified NSGA-II algorithm
population = nsga2_algorithm(num_generations, pop_size, num_cities, num_items, distance_matrix, item_values)

