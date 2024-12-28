import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
from itertools import permutations

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def parse_ttp_instance(filepath):
    """
    解析 TTP 实例文件（如 a280-TTP）。
    返回:
      - num_cities: 城市数量 (不含起点重复)
      - num_items: 物品数量
      - capacity: 背包容量 Q
      - v_min, v_max: 最小、最大速度
      - renting_ratio: 租用比率(若不考虑则可忽略)
      - coords: [(x1, y1), (x2, y2), ...] 形状的城市坐标
      - items: [(profit_j, weight_j, assigned_city_j), ...]  (价值，重量，所属城市)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_cities = None
    num_items = None
    capacity = None
    v_min = None
    v_max = None
    renting_ratio = None
    coords = []
    items = []

    # 简易解析（注意：不同 TTP 文件格式稍有差异，需按需修改）
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("DIMENSION"):
            # e.g. DIMENSION: 280
            num_cities = int(line.split()[-1])
        elif line.startswith("NUMBER OF ITEMS"):
            # e.g. NUMBER OF ITEMS: 279
            num_items = int(line.split()[-1])
        elif line.startswith("CAPACITY OF KNAPSACK"):
            capacity = int(line.split()[-1])
        elif line.startswith("MIN SPEED"):
            v_min = float(line.split()[-1])
        elif line.startswith("MAX SPEED"):
            v_max = float(line.split()[-1])
        elif line.startswith("RENTING RATIO"):
            renting_ratio = float(line.split()[-1])
        elif line.startswith("NODE_COORD_SECTION"):
            # 读取城市坐标
            j = i + 1
            while not lines[j].startswith("ITEMS SECTION"):
                xytoken = lines[j].strip().split()
                # xytoken 形如: [index, x, y]
                x = float(xytoken[1])
                y = float(xytoken[2])
                coords.append((x, y))
                j += 1
            i = j - 1  # 调整 i
        elif line.startswith("ITEMS SECTION"):
            # 解析 items
            j = i + 1
            while j < len(lines):
                if not lines[j].strip():
                    j += 1
                    continue
                tokens = lines[j].strip().split()
                # 形如: [index, profit, weight, assignedCity]
                if len(tokens) < 4:
                    break
                # 不一定需 index, 但可以做个记录
                profit_j = float(tokens[1])
                weight_j = float(tokens[2])
                city_j = int(tokens[3])  # 物品所属于的城市编号(1-based)
                items.append((profit_j, weight_j, city_j))
                j += 1
            i = j - 1
        i += 1

    return num_cities, num_items, capacity, v_min, v_max, renting_ratio, coords, items


def compute_dist_matrix(coords):    #计算每个城市之间的距离 并用矩阵表示
    """
    根据城市坐标计算距离矩阵 dist[i][j].
    假设使用欧氏距离或 CEIL_2D (向上取整).
    """
    n = len(coords)
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0.0
            else:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                d = math.sqrt(dx*dx + dy*dy)
                dist[i][j] = math.ceil(d)  # CEIL_2D
    return dist


def evaluate_solution(sol, dist_matrix, items, capacity, v_min, v_max):     #评估一个解决方案的质量
    """
    评估一个 TTP 解的目标值 (time, profit).
    sol = (route, picked) 其中:
      - route: list of city indices, 形如 [0, 1, 2, ..., n-1],
               表示从城市0出发, 然后1,2,...,最后回到0 (这里可自己决定是否显式回到0)
      - picked: set 或 list, 表示选了哪些物品的 '全局索引'
                或者: { item_index1, item_index2, ... }
                (你也可以改成以城市为维度记录 picked)
    注意：城市编号如果在文件中是 1-based, 内部可以减1 做处理。

    返回:
      time -> float (要最小化)
      profit -> float (要最大化)
    """

    # 1. 计算路途中背包重量变化
    #    我们需要知道在 route 的每一步，会累积多少背包重量
    #    先把 items 按 city 分组，方便快速查找
    route, picked = sol
    city2items = {}
    for i_idx, (p, w, c) in enumerate(items):
        # c-1 因为城市是1-based
        city2items.setdefault(c-1, []).append((i_idx, p, w))    #把对应物品放到对应下标的数组下吗

    total_profit = 0.0
    cur_weight = 0.0
    total_time = 0.0

    # 遍历 route 的相邻城市对
    n = len(route)
    for i in range(n):
        c1 = route[i]   #第一个城市下标
        c2 = route[(i+1) % n]  # 回到首城 (典型TSP循环)
        # 在 c1 城市，若有可选物品，则检查 picked 是否选择
        if c1 in city2items:
            for (item_idx, p, w) in city2items[c1]: #找到城市对应物品
                if picked[item_idx] == 1:
                    # 如果选了这个物品
                    # 检查背包容量
                    if cur_weight + w <= capacity:  #按说能进行质量评估的解都已经判断好了不会超出最大重量限制//////////////
                        cur_weight += w
                        total_profit += p
                    else:
                        # 容量超限，可视需求：要么忽略，要么判 infeasible
                        # 这里简单地直接忽略
                        pass

        # 计算从 c1 到 c2 的路程, dist_matrix[c1][c2]
        # 根据当前背包重量, 计算速度
        w_ratio = cur_weight / float(capacity)
        # v(w) = v_max - w_ratio*(v_max - v_min)
        if w_ratio > 1.0:
            # 背包超限时，这里可做惩罚(或速度=v_min)
            v_current = v_min
        else:
            v_current = v_max - w_ratio*(v_max - v_min)

        distance = dist_matrix[c1][c2]
        travel_time = distance / v_current
        total_time += travel_time

    return total_time, total_profit


def dominates(fitA, fitB):
    """
    fitA = (timeA, profitA)
    fitB = (timeB, profitB)
    TTP中: 希望 time 越小越好, profit 越大越好
    A 支配 B 的条件:
      timeA <= timeB AND profitA >= profitB
      且至少有一个严格 ( < 或 > )
    """
    (timeA, profA) = fitA
    (timeB, profB) = fitB
    cond1 = (timeA <= timeB)
    cond2 = (profA >= profB)
    strict = (timeA < timeB) or (profA > profB)
    return cond1 and cond2 and strict


def fast_non_dominated_sort(pop_fits):  #帕累托排序
    """
    NSGA-II 快速非支配排序
    pop_fits: list of fitness, e.g. [(time1, profit1), (time2, profit2), ...]
    返回:
      fronts: list of lists, 每层 front 是若干下标
    """
    S = [[] for _ in range(len(pop_fits))]
    n = [0]*len(pop_fits)   # n[i] = 支配 i 的数量
    fronts = [[]]

    for p in range(len(pop_fits)):
        for q in range(len(pop_fits)):
            if p == q:
                continue
            if dominates(pop_fits[p], pop_fits[q]):
                S[p].append(q)
            elif dominates(pop_fits[q], pop_fits[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    # 移除空的最后一层
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(front, pop_fits): #拥挤距离
    """
    计算 NSGA-II 中的拥挤距离
    front: 同一层中的若干个体下标
    pop_fits: [(time, profit), ...]
    返回: { idx: distance }
    """
    distance = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distance[idx] = float('inf')
        return distance

    # 分别对 time 和 profit 排序
    # time 越小越好
    front_sorted_time = sorted(front, key=lambda x: pop_fits[x][0]) #根据时间排序（从小到大）
    # profit 越大越好 => 排序时取负值或 reverse
    front_sorted_profit = sorted(front, key=lambda x: pop_fits[x][1], reverse=False)    #根据价值排序（从小到大）

    # time range
    time_min = pop_fits[front_sorted_time[0]][0]    #时间最小的时间
    time_max = pop_fits[front_sorted_time[-1]][0]   #时间最大的时间
    time_range = time_max - time_min if time_max != time_min else 1e-9

    # profit range
    prof_min = pop_fits[front_sorted_profit[0]][1]  #价值最小的价值
    prof_max = pop_fits[front_sorted_profit[-1]][1] #价值最大的价值
    prof_range = prof_max - prof_min if prof_max != prof_min else 1e-9

    # 边界个体置为 inf
    distance[front_sorted_time[0]] = float('inf')
    distance[front_sorted_time[-1]] = float('inf')
    distance[front_sorted_profit[0]] = float('inf')
    distance[front_sorted_profit[-1]] = float('inf')

    # time 维度
    for i in range(1, len(front) - 1):
        idx_before = front_sorted_time[i - 1]   #小些时间的对应下标
        idx_after  = front_sorted_time[i + 1]   #大些时间的对应下标
        idx_curr   = front_sorted_time[i]   #当前时间下标 适中大小
        dist_part  = (pop_fits[idx_after][0] - pop_fits[idx_before][0]) / time_range
        distance[idx_curr] += dist_part

    # profit 维度
    for i in range(1, len(front) - 1):
        idx_before = front_sorted_profit[i - 1] #小些价值的对应下标
        idx_after  = front_sorted_profit[i + 1] #大些价值的对应下标
        idx_curr   = front_sorted_profit[i] #当前价值的下标 适中大小
        dist_part  = (pop_fits[idx_after][1] - pop_fits[idx_before][1]) / prof_range
        distance[idx_curr] += dist_part

    return distance #返回front中每个下标对应的距离值


def tournament_selection(population, fits, k=3):
    """
    简单锦标赛选择，返回赢家个体
    population: list of (solution, fitness) or we store them separately
    fits: same order as population
    k: tournament size
    """
    selected = random.sample(range(len(population)), k)
    best_idx = selected[0]
    for idx in selected[1:]:
        if dominates(fits[idx], fits[best_idx]):
            best_idx = idx
    return population[best_idx], fits[best_idx]


def order_crossover(routeA, routeB):    #route两点交叉
    """
    对 TSP 路径进行一个常见的OX交叉(可自行换别的).
    返回 child route
    """
    n = len(routeA)
    start, end = sorted(random.sample(range(n), 2))
    child = [None]*n
    child[start:end] = routeA[start:end]
    posB = 0
    for i in range(n):
        if routeB[i] not in child:
            while child[posB] is not None:
                posB += 1
            child[posB] = routeB[i]
    return child


def bitwise_mutation(bit_list, pm=0.01):    #物品选择突变  以[1, 0, 1, 1, 0, 1]形式表示
    """
    对物品选择进行简单位翻转
    """
    for i in range(len(bit_list)):
        if random.random() < pm:
            bit_list[i] = 1 - bit_list[i]
    return bit_list


def swap_mutation(route, pm=0.02):  #route突变
    """
    对路径进行swap突变
    """
    n = len(route)
    for i in range(n):
        if random.random() < pm:
            j = random.randint(0, n-1)
            route[i], route[j] = route[j], route[i]
    return route


def nsga2_main(dist_matrix, items, capacity, v_min, v_max, pop_size, n_gen, p_c, p_m):
    """
    NSGA-II 主流程
    返回: 最终的 (population, fits)
    """
    num_cities = len(dist_matrix)
    num_items = len(items)

    # ========== 初始化种群 ==========
    population = [] #种群
    fits = []   #对应的适应度

    def random_individual():
        # 路径: 随机排列 [0, 1, 2, ..., n-1]
        route = list(range(num_cities))
        random.shuffle(route)
        # 物品选择: 长度 = num_items, 0/1
        # 也可以更严格：只给 route[1..n-1]城市所在物品才有可能1
        picking = [random.randint(0,1) for _ in range(num_items)]   #可能会超出背包最大重量限制
        return (route, picking)

    # 生成 pop_size 个随机解
    for _ in range(pop_size):
        sol = random_individual()
        # 评估
        time_val, profit_val = evaluate_solution(sol, dist_matrix, items, capacity, v_min, v_max)
        population.append(sol)
        fits.append((time_val, profit_val))

    # ========== 开始迭代 ==========
    for gen in range(n_gen):
        # 生成子代(交叉+变异)
        offspring = []
        offspring_fits = []

        while len(offspring) < pop_size:
            # 1) 选择
            p1, f1 = tournament_selection(population, fits) #如果互相不支配的话 是否考虑对比拥挤距离//////////////
            p2, f2 = tournament_selection(population, fits)

            # 2) 交叉 (对 route 和 picking 分开处理)
            route1, pick1 = copy.deepcopy(p1)
            route2, pick2 = copy.deepcopy(p2)

            if random.random() < p_c:   #交叉概率
                child_route = order_crossover(route1, route2)
            else:
                child_route = copy.deepcopy(route1)

            # 物品交叉(可简化成单点交叉或直接选择其一)
            if random.random() < p_c:
                cut_point = random.randint(0, num_items-1)
                child_pick = pick1[:cut_point] + pick2[cut_point:]
            else:
                child_pick = copy.deepcopy(pick1)

            # 3) 变异
            child_route = swap_mutation(child_route, pm=p_m)
            child_pick = bitwise_mutation(child_pick, pm=p_m)

            child_sol = (child_route, child_pick)

            # 4) 评估
            c_time, c_profit = evaluate_solution(
                child_sol,
                dist_matrix,
                items,
                capacity,
                v_min,
                v_max
            )
            offspring.append(child_sol)
            offspring_fits.append((c_time, c_profit))

        # ========== 环境选择 (NSGA-II) ==========
        # 合并
        combined_pop = population + offspring
        combined_fits = fits + offspring_fits

        # 非支配排序
        fronts = fast_non_dominated_sort(combined_fits)

        new_pop = []    #要替换父代的新子代
        new_fits = []   #对应的质量
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                # 整个 front 全部加入
                for idx in front:
                    new_pop.append(combined_pop[idx])
                    new_fits.append(combined_fits[idx])
            else:
                # 需要基于拥挤距离挑选一部分
                cd = crowding_distance(front, combined_fits)
                # 按距离从大到小排序
                front_sorted = sorted(front, key=lambda x: cd[x], reverse=True) #从大到小排序
                needed = pop_size - len(new_pop)
                chosen = front_sorted[:needed]
                for idx in chosen:
                    new_pop.append(combined_pop[idx])
                    new_fits.append(combined_fits[idx])
                break

        population = new_pop
        fits = new_fits

        if (gen+1) % 10 == 0:
            # print(f"Gen {gen+1}/{n_gen}: best (time, profit) so far?")
            fronts = fast_non_dominated_sort(fits)
            rank0 = fronts[0]
            print(f"Gen {gen+1}/{n_gen} - rank0 solutions:")

            # 计算 rank0 层中时间和价值的平均值
            total_time = 0.0
            total_profit = 0.0
            for idx in rank0:
                t, p = fits[idx]
                total_time += t
                total_profit += p
                # print(f"  idx={idx}, time={t:.2f}, profit={p:.2f}")
            # 这里可展示一下最优或平均
            # 例如按照 time + (-profit) 排序只是单参考
            # 你可以做更多监控
            size_rank0 = len(rank0)
            avg_time = total_time / size_rank0 if size_rank0 > 0 else 0.0
            avg_profit = total_profit / size_rank0 if size_rank0 > 0 else 0.0

            print(f"  => rank0_size={size_rank0}, avg_time={avg_time:.2f}, avg_profit={avg_profit:.2f}\n")
            pass

    return population, fits


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


def main():
    # ========== 读取并解析 TTP 实例 (以 a280-TTP 为例) ==========
    ttp_file = "./fnl4461-n22300.txt"
    (num_cities, num_items, capacity,
     v_min, v_max, renting_ratio, coords, items) = parse_ttp_instance(ttp_file)

    set_random_seed(42)
    # 计算距离矩阵
    dist_matrix = compute_dist_matrix(coords)

    # 运行 NSGA-II
    final_pop, final_fits = nsga2_main(
        dist_matrix, items, capacity, v_min, v_max,
        pop_size=100,  # 可调
        n_gen=100,    # 可调
        p_c=0.9,
        p_m=0.02
    )

    # ========== 结果分析 ==========
    # 可以对 final_fits 做一次非支配排序，获取帕累托前沿
    fronts = fast_non_dominated_sort(final_fits)
    for fon in fronts[0]:
        print('rank0:', final_fits[fon])
    plot_pareto_front(final_fits, fronts)
    # print("\nFinal Pareto fronts (index-based):")
    # for i, front in enumerate(fronts):
    #     print(f"  Front {i}: {front}")

    # # 也可以打印具体 (time, profit) 看看
    # print("\nSome solutions on the first front:")
    # first_front = fronts[0]
    # for idx in first_front[:5]:  # 仅举例打印
    #     sol = final_pop[idx]
    #     ft = final_fits[idx]
    #     print(f"  idx={idx}, route={sol[0][:10]}..., picked_count={sum(sol[1])}, time={ft[0]:.2f}, profit={ft[1]:.2f}")






if __name__ == "__main__":
    main()
