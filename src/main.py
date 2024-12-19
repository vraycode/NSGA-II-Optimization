import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Load the problem data
def load_problem(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    cities = []     # 城市数组
    items = []      # 物品数组
    capacity = 0        # 背包容量
    min_speed = 0       # 最小速度
    max_speed = 0       # 最大速度
    renting_ratio = 0       # 目标间权衡参数
    for line in data:
        if line.startswith("CAPACITY OF KNAPSACK"):     #背包容量
            capacity = int(line.split(':')[-1])
        elif line.startswith("MIN SPEED"):      #最小速度
            min_speed = float(line.split(':')[-1])
        elif line.startswith("MAX SPEED"):      #最大速度
            max_speed = float(line.split(':')[-1])
        elif line.startswith("RENTING RATIO"):      #加权公式
            renting_ratio = float(line.split(':')[-1])
        elif line.startswith("NODE_COORD_SECTION"):
            idx = data.index(line) + 1
            while not data[idx].startswith("ITEMS SECTION"):
                parts = data[idx].split()
                cities.append((int(parts[0]), float(parts[1]), float(parts[2])))
                idx += 1
        elif line.startswith("ITEMS SECTION"):
            idx = data.index(line) + 1
            while idx < len(data):
                parts = data[idx].split()
                items.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))      #下标、价值、重量、所在城市
                idx += 1
    return cities, items, capacity, min_speed, max_speed, renting_ratio

# Initialize the population
def initialize_population(pop_size, num_cities, num_items, items, capacity):
    population = []
    for _ in range(pop_size):
        tour = random.sample(range(1, num_cities + 1), num_cities)  # 随机生成路径
        packing_plan = [0] * num_items  # 初始化物品选择方案
        total_weight = 0  # 当前背包重量

        # 按照路径顺序选择物品
        for city in tour:
            for j, item in enumerate(items):
                if item[3] == city and total_weight + item[2] <= capacity:  # 判断物品是否属于当前城市且不超重 保证了先到达的城市中的物品有更大概率被选择
                    if random.random() < 0.5:  # 50%的概率选择该物品
                        packing_plan[j] = 1
                        total_weight += item[2]
        population.append((tour, packing_plan))
    return population

# Fitness function: calculates time and profit
def evaluate_solution(solution, cities, items, capacity, min_speed, max_speed, renting_ratio):
    tour, packing_plan = solution       #每个解的旅行路径和物品选择
    total_weight = 0        #总重量
    total_profit = 0        #总价值
    total_time = 0      #总时间

    # 计算背包初始重量和总收益
    for i, pick in enumerate(packing_plan):
        if pick == 1:
            total_profit += items[i][1]     #价值
            total_weight += items[i][2]     #重量

    current_weight = 0  # 初始小偷背包重量
    for i in range(len(tour)):      #遍历旅行路径
        city1 = cities[tour[i] - 1]     #要对应下标，所以减1
        city2 = cities[tour[(i + 1) % len(tour)] - 1]

        # 更新背包当前重量：在当前城市拾取物品
        for j, item in enumerate(items):    #次循环中没有加break防止一个城市有多个物品
            if item[3] == tour[i] and packing_plan[j] == 1:     #先判断目前这个城市对应的哪个物品  然后判断此物品有没有被选择
                current_weight += item[2]
        
        # 根据当前背包重量计算速度 v
        current_speed = max(max_speed - (current_weight / capacity) * (max_speed - min_speed), min_speed)

        # 计算从 city1 到 city2 的距离
        dist = distance.euclidean((city1[1], city1[2]), (city2[1], city2[2]))       #根据两城市的x，y计算两点距离

        # 累加当前城市到下一城市的旅行时间
        total_time += dist / current_speed

    # 计算总时间成本
    total_cost = renting_ratio * total_time

    return total_cost, total_profit


# Non-dominated sorting     非支配排序
def non_dominated_sorting(population, fitness_values):
    fronts = []
    domination_count = [0] * len(population)  #记录每个解被支配的次数
    dominated_solutions = [[] for _ in range(len(population))]      #记录每个解支配的其他解

    for p in range(len(population)):    #对比第一个
        for q in range(len(population)):
            if ((fitness_values[p][0] <= fitness_values[q][0] and fitness_values[p][1] > fitness_values[q][1]) or \
            (fitness_values[p][0] < fitness_values[q][0] and fitness_values[p][1] >= fitness_values[q][1])):
                dominated_solutions[p].append(q)    #p支配的解
            elif ((fitness_values[q][0] <= fitness_values[p][0] and fitness_values[q][1] > fitness_values[p][1]) or \
            (fitness_values[q][0] < fitness_values[p][0] and fitness_values[q][1] >= fitness_values[p][1])):
                domination_count[p] += 1    #p被支配

        if domination_count[p] == 0:  #证明是rank0
            if len(fronts) == 0:
                fronts.append([])
            fronts[0].append(p)

    i = 0
    if len(fronts) == 0:  # 确保fronts有一个初始的非空层
        fronts.append([])  
    while i < len(fronts) and len(fronts[i]) > 0:  # 确保不会越界访问
        next_front = []
        for p in fronts[i]:     #rank0
            for q in dominated_solutions[p]:    #遍历p支配的解  如果rank0中有2个解同时支配了一个解
                domination_count[q] -= 1    #这一次就是被p支配的一次
                if domination_count[q] == 0:
                    next_front.append(q)
        if len(next_front) > 0:  # 避免添加空的层
            fronts.append(next_front)
        # fronts.append(next_front)
        i += 1

    return fronts

# Crowding distance calculation
def calculate_crowding_distance(front, fitness_values):     #一层帕累托前沿解集和整个fitness_values
    distances = [0] * len(front)        #初始化每个的距离值为0
    num_objectives = len(fitness_values[0])     #双目标问题

    for m in range(num_objectives):
        sorted_indices = sorted(range(len(front)), key=lambda i: fitness_values[front[i]][m])       #例如根据代价大小把对应下标从小到大排序
        distances[sorted_indices[0]] = float('inf')     #代价最小对应距离值设为无穷大
        distances[sorted_indices[-1]] = float('inf')    #代价最大对应距离值设为无穷大

        for i in range(1, len(front) - 1):
            distances[sorted_indices[i]] += (fitness_values[front[sorted_indices[i + 1]]][m] - fitness_values[front[sorted_indices[i - 1]]][m])

    return distances

# Selection: Tournament selection 锦标赛选择  k=2 选择压力较小
def tournament_selection(population, fitness_values, ranks, crowding_distances, k):
    candidates = random.sample(range(len(population)), k)       #随机选两个
    best_candidate = candidates[0]
    for candidate in candidates[1:]:
        if ranks[candidate] < ranks[best_candidate]:    #对比等级
            best_candidate = candidate
        elif ranks[candidate] == ranks[best_candidate]:     #对比距离
            if crowding_distances[candidate] > crowding_distances[best_candidate]:
                best_candidate = candidate

    return population[best_candidate]

# Crossover operation  交叉
def crossover(parent1, parent2, items, capacity):
    tour1, packing_plan1 = parent1
    tour2, packing_plan2 = parent2
    # TSP crossover: Order Crossover (OX)  成本目标进行交叉
    size = len(tour1)
    start, end = sorted(random.sample(range(size), 2))  #二点交叉
    child_tour = [-1] * size
    child_tour[start:end] = tour1[start:end]
    for city in tour2:
        if city not in child_tour:
            child_tour[child_tour.index(-1)] = city
    # Knapsack crossover: Uniform Crossover   价值目标进行交叉
    # child_packing_plan = [random.choice([g1, g2]) for g1, g2 in zip(packing_plan1, packing_plan2)]
    child_packing_plan = [0] * len(packing_plan1)   #交叉操作加入是否超过最大背包限制判断 实现根据访问城市顺序来选择物品
    total_weight = 0
    for city in child_tour:
        for j, item in enumerate(items):
            if item[3] == city and total_weight + item[2] <= capacity:
                choice = random.choice([packing_plan1[j], packing_plan2[j]])
                if choice == 1 and total_weight + item[2] <= capacity:  # 按路径顺序选择
                    child_packing_plan[j] = 1
                    total_weight += item[2]

    return (child_tour, child_packing_plan)

# Mutation operation  突变
def mutate(solution, mutation_rate, items, capacity):       #目前突变率设置为0.1
    tour, packing_plan = solution
    # TSP mutation: Swap two cities
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)       #两点突变
        tour[i], tour[j] = tour[j], tour[i]
    # Knapsack mutation: Flip bit  加入是否超出背包重量限制判断
    total_weight = 0  # 重置当前背包重量
    new_packing_plan = [0] * len(items)  # 初始化新的物品选择计划

    for city in tour:  # 按路径顺序遍历城市   先访问的城市对应物品先进行突变
        for j, item in enumerate(items):
            if item[3] == city:  # 判断物品是否属于当前访问的城市
                if random.random() < mutation_rate:
                    # 尝试进行突变：翻转选择状态，判断是否超重
                    new_pick = 1 - packing_plan[j]
                    if new_pick == 1 and total_weight + item[2] <= capacity:
                        new_packing_plan[j] = 1
                        total_weight += item[2]
                    else:
                        new_packing_plan[j] = 0
                else:
                     # 尝试保持原状态，但需验证合法性
                    if packing_plan[j] == 1 and total_weight + item[2] <= capacity:
                        new_packing_plan[j] = 1
                        total_weight += item[2]
                    else:
                        new_packing_plan[j] = 0  # 强制修改为未选择
    
    return (tour, new_packing_plan)

# Main NSGA-II algorithm
def nsga2(cities, items, capacity, min_speed, max_speed, renting_ratio, pop_size, generations):
    population = initialize_population(pop_size, len(cities), len(items), items, capacity)       #生成初始解集[(城市访问路径，物品选择状况),...]
    for gen in range(generations):      #规定迭代次数
        fitness_values = [evaluate_solution(ind, cities, items, capacity, min_speed, max_speed, renting_ratio) for ind in population]       #评估每一个解的质量[（总时间,总价值）,...]
        fronts = non_dominated_sorting(population, fitness_values)      #当前解集 非支配排序（帕累托排序）

        # Assign ranks and calculate crowding distances
        ranks = [0] * len(population)
        for i, front in enumerate(fronts):
            for ind in front:
                ranks[ind] = i      #给每个解设置对应的rank等级

        crowding_distances = [0] * len(population)
        for front in fronts:
            front_distances = calculate_crowding_distance(front, fitness_values)
            for i, ind in enumerate(front):
                crowding_distances[ind] = front_distances[i]        #给每个解设置对应的距离

        # Generate offspring  生成子代 N个
        offspring = []
        while len(offspring) < pop_size:
            parent1 = tournament_selection(population, fitness_values, ranks, crowding_distances, 3) #选择压力为2
            parent2 = tournament_selection(population, fitness_values, ranks, crowding_distances, 3)
            child = crossover(parent1, parent2, items, capacity)     #交叉 防止产生无效解的判断
            child = mutate(child, 0.1, items, capacity)      #突变  防止产生无效解的判断
            offspring.append(child)     #后代生成

        # Combine population and offspring   结合父代与子代  2N
        combined_population = population + offspring
        fitness_values = [evaluate_solution(ind, cities, items, capacity, min_speed, max_speed, renting_ratio) for ind in combined_population]  #为2N生成适应度
        fronts = non_dominated_sorting(combined_population, fitness_values) #2N的非支配排序

        #更新下一代父代
        new_population = []
        for front in fronts:        #帕累托排序中的每一层
            if len(new_population) + len(front) <= pop_size:    #如果都加上也不会超出限制
                new_population.extend([combined_population[ind] for ind in front])
            else:       #如果一层都加上会超出限制 这需要进行筛选
                front_distances = calculate_crowding_distance(front, fitness_values)    #帕累托排序中给每一层中每个解的拥挤距离值
                sorted_front = sorted(zip(front, front_distances), key=lambda x: x[1], reverse=True)    #根据距离降序排序
                for ind, _ in sorted_front:
                    new_population.append(combined_population[ind])
                    if len(new_population) == pop_size:
                        break
            if len(new_population) == pop_size:
                break

        population = new_population
        

    return population

# Example usage
filename = "./a280-n279.txt"  # Replace with your input file path
cities, items, capacity, min_speed, max_speed, renting_ratio = load_problem(filename)
result = nsga2(cities, items, capacity, min_speed, max_speed, renting_ratio, 100, 200)      #最后生成的200个解

fitness_values = [evaluate_solution(ind, cities, items, capacity, min_speed, max_speed, renting_ratio) for ind in result]       #100个解的质量[（总时间,总价值）,...]
fronts = non_dominated_sorting(result, fitness_values)      #当前解集 非支配排序（帕累托排序）

pareto_front = []   #帕累托前沿rank0的解质量
for i in fronts[0]:
    pareto_front.append(fitness_values[i])
print(pareto_front)

def plot_pareto_front(fitness_values, fronts):
    """
    绘制帕累托前沿可视化图。
    
    Parameters:
        fitness_values: 列表，包含所有解的目标值 [(total_cost, total_profit), ...]
        fronts: 帕累托排序后的解的索引列表 [ [rank0解索引], [rank1解索引], ...]
    """
    # 为不同rank的帕累托层分配颜色
    colors = plt.cm.get_cmap('tab10', len(fronts))  # 使用tab10颜色

    plt.figure(figsize=(10, 6))
    for rank, front in enumerate(fronts):
        # 提取当前rank的目标值
        rank_values = [fitness_values[i] for i in front]
        costs = [v[0] for v in rank_values]  # X轴：总时间
        profits = [v[1] for v in rank_values]  # Y轴：总价值
        
        # 绘制散点图
        plt.scatter(costs, profits, color=colors(rank), label=f'Rank {rank}', s=50)

    plt.xlabel('Total Time (Cost)')
    plt.ylabel('Total Profit')
    plt.title('Pareto Front Visualization')
    plt.legend()
    plt.grid()
    plt.show()

# 假设已经有 fitness_values 和 fronts
# fitness_values = [(cost1, profit1), (cost2, profit2), ...]
# fronts = [[索引1, 索引2], [索引3, 索引4], ...]  # 帕累托排序结果

# plot_pareto_front(fitness_values, fronts)








# def calculate_hypervolume(front, reference_point):
#     # Sort the front by the first objective (ascending)
#     front = sorted(front, key=lambda x: x[0])

#     hv = 0.0
#     prev_f1 = reference_point[0]
#     for f1, f2 in front:
#         width = prev_f1 - f1
#         height = reference_point[1] - f2
#         hv += width * height
#         prev_f1 = f1

#     return hv

# Example Usage
# reference_point = [5444.0, -0.0]  # Reference point
# hypervolume = calculate_hypervolume(pareto_front, reference_point)
# print("Hypervolume:", hypervolume)

