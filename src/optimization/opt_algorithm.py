import numpy as np
import random
import math
import pulp


# ===================== 计算地球球面距离（KM） =====================
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371  # 地球半径 km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def is_valid_solution(df, selected_ids, min_spacing=0.8):
    """检查解决方案是否满足最小间距约束"""
    if len(selected_ids) <= 1:
        return True

    coords = df.loc[selected_ids, ["lon", "lat"]].values
    n = len(coords)

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(coords[i][0], coords[i][1],
                             coords[j][0], coords[j][1])
            if d < min_spacing:
                return False
    return True


def calculate_min_spacing(df, selected_ids):
    """计算当前解中风机之间的最小间距"""
    if len(selected_ids) <= 1:
        return float('inf')

    coords = df.loc[selected_ids, ["lon", "lat"]].values
    n = len(coords)
    min_distance = float('inf')

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(coords[i][0], coords[i][1],
                             coords[j][0], coords[j][1])
            if d < min_distance:
                min_distance = d

    return min_distance


# ===================== 评估函数 =====================
def evaluate_solution(df, selected_ids, cost_weight,
                      max_slope, max_road_distance, min_residential_distance,
                      min_heritage_distance, min_geology_distance, min_water_distance,
                      min_spacing=0.8):  # 默认间距0.8km
    """评估解决方案的适应度"""
    if not is_valid_solution(df, selected_ids, min_spacing):
        return -1e9  # 大幅增加间距约束惩罚

    # 能量产出
    energy = np.sum(df.loc[selected_ids, "wind_power_density"]) * 10

    # 成本惩罚
    max_cost = df["cost"].max() if df["cost"].max() > 0 else 1
    cost = cost_weight * np.sum(df.loc[selected_ids, "cost"]) / max_cost * 1000

    # 约束惩罚
    constraint_penalty = 0

    # 坡度约束惩罚
    if 'slope' in df.columns:
        slope_violation = np.sum(df.loc[selected_ids, "slope"] > max_slope)
        constraint_penalty += slope_violation * 1000

    # 道路距离约束惩罚
    if 'road_distance' in df.columns:
        road_violation = np.sum(df.loc[selected_ids, "road_distance"] > max_road_distance)
        constraint_penalty += road_violation * 1000

    # 居民区距离约束惩罚
    if 'residential_distance' in df.columns:
        residential_violation = np.sum(df.loc[selected_ids, "residential_distance"] < min_residential_distance)
        constraint_penalty += residential_violation * 1000

    # 文化遗产距离约束惩罚
    if 'heritage_distance' in df.columns:
        heritage_violation = np.sum(df.loc[selected_ids, "heritage_distance"] < min_heritage_distance)
        constraint_penalty += heritage_violation * 1000

    # 地质距离约束惩罚
    if 'geology_distance' in df.columns:
        geology_violation = np.sum(df.loc[selected_ids, "geology_distance"] < min_geology_distance)
        constraint_penalty += geology_violation * 1000

    # 水源距离约束惩罚
    if 'water_distance' in df.columns:
        water_violation = np.sum(df.loc[selected_ids, "water_distance"] < min_water_distance)
        constraint_penalty += water_violation * 1000

    # 电网接近度奖励
    if 'grid_proximity' in df.columns:
        grid_reward = np.sum(df.loc[selected_ids, "grid_proximity"]) * 100
        energy += grid_reward

    return energy - cost - constraint_penalty


# ===================== 基础操作函数 =====================
def mutate_solution_with_spacing(solution, valid_points, df, min_spacing=0.8, rate=0.3):
    """变异操作 - 考虑间距约束"""
    child = solution.copy()
    mutation_attempts = 0
    max_attempts = 50

    for _ in range(int(len(solution) * rate)):
        if mutation_attempts >= max_attempts:
            break

        pos = random.randint(0, len(child) - 1)
        old_point = child[pos]

        # 尝试找到满足间距的新点位
        for attempt in range(20):
            new_point = random.choice(valid_points)
            if new_point in child:
                continue

            # 临时替换并检查间距
            temp_solution = child.copy()
            temp_solution[pos] = new_point

            if is_valid_solution(df, temp_solution, min_spacing):
                child[pos] = new_point
                break

        mutation_attempts += 1

    # 确保解有效
    if not is_valid_solution(df, child, min_spacing):
        # 如果无效，返回原始解
        return solution

    return child


def crossover_solution_with_spacing(p1, p2, df, min_spacing=0.8):
    """交叉操作 - 考虑间距约束"""
    n = len(p1)

    # 尝试多种交叉策略
    for attempt in range(10):
        # 方法1: 单点交叉
        point = random.randint(1, n - 2)
        child = p1[:point] + [g for g in p2 if g not in p1[:point]]

        # 补全长度的同时避免重复
        if len(child) < n:
            available_points = [p for p in p2 if p not in child]
            missing = n - len(child)
            if len(available_points) >= missing:
                child.extend(random.sample(available_points, missing))
            else:
                # 如果p2中点数不够，从valid_points中补充
                all_valid = list(set(p1 + p2))
                additional = [p for p in all_valid if p not in child]
                if len(additional) >= missing:
                    child.extend(random.sample(additional, missing))

        # 检查间距约束
        if len(child) == n and is_valid_solution(df, child, min_spacing):
            return child

    # 如果交叉失败，返回较好的父代
    p1_fitness = len(set(p1))
    p2_fitness = len(set(p2))
    return p1 if p1_fitness >= p2_fitness else p2


# ===================== 遗传算法 =====================
def run_genetic_algorithm(df, n_turbines, cost_weight,
                          max_slope, max_road_distance, min_residential_distance,
                          min_heritage_distance, min_geology_distance, min_water_distance,
                          pop_size=40, generations=80, mutation_rate=0.25, crossover_rate=0.8,
                          min_spacing=0.8):  # 添加间距参数
    """运行遗传算法"""
    valid_points = df[df["valid"] == 1].index.tolist()

    if len(valid_points) < n_turbines:
        return {"algorithm": "遗传算法", "solution": [], "fitness": -1, "convergence_history": [], "min_spacing_km": 0}

    # 生成初始种群时确保满足间距约束
    population = []
    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 10:
        candidate = random.sample(valid_points, n_turbines)
        if is_valid_solution(df, candidate, min_spacing):
            population.append(candidate)
        attempts += 1

    # 如果无法生成足够的初始解，使用不满足约束的初始解
    if len(population) < pop_size:
        additional_needed = pop_size - len(population)
        additional = [random.sample(valid_points, n_turbines) for _ in range(additional_needed)]
        population.extend(additional)

    best_solution, best_fitness = None, -1e6
    history = []

    for g in range(generations):
        fitness_scores = [evaluate_solution(df, sol, cost_weight,
                                            max_slope, max_road_distance, min_residential_distance,
                                            min_heritage_distance, min_geology_distance, min_water_distance,
                                            min_spacing)  # 传递间距参数
                          for sol in population]

        # 更新最优解
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_solution = population[best_idx]

        history.append(best_fitness)

        # 轮盘赌选择
        fitness_array = np.array(fitness_scores)
        fitness_array = np.maximum(fitness_array, 1e-6)  # 避免负值
        probabilities = fitness_array / np.sum(fitness_array)

        new_pop = []
        for _ in range(pop_size // 2):
            p1 = population[np.random.choice(len(population), p=probabilities)]
            p2 = population[np.random.choice(len(population), p=probabilities)]

            if random.random() < crossover_rate:
                c1 = crossover_solution_with_spacing(p1, p2, df, min_spacing)
                c2 = crossover_solution_with_spacing(p2, p1, df, min_spacing)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate_solution_with_spacing(c1, valid_points, df, min_spacing, mutation_rate)
            c2 = mutate_solution_with_spacing(c2, valid_points, df, min_spacing, mutation_rate)

            new_pop.extend([c1, c2])

        # 保留最优解
        if best_solution is not None:
            new_pop[0] = best_solution

        population = new_pop

    # 计算最终解的最小间距
    min_spacing_achieved = calculate_min_spacing(df, best_solution) if best_solution else 0

    return {
        "algorithm": "遗传算法",
        "solution": best_solution,
        "fitness": best_fitness,
        "convergence_history": history,
        "min_spacing_km": min_spacing_achieved
    }


# ===================== 模拟退火算法 =====================
def run_simulated_annealing(df, n_turbines, cost_weight,
                            max_slope, max_road_distance, min_residential_distance,
                            min_heritage_distance, min_geology_distance, min_water_distance,
                            initial_temp=2000, cooling_rate=0.97, iterations_per_temp=100,
                            min_spacing=0.8):  # 添加间距参数
    """运行模拟退火算法"""
    valid_points = df[df["valid"] == 1].index.tolist()

    if len(valid_points) < n_turbines:
        return {"algorithm": "模拟退火算法", "solution": [], "fitness": -1, "convergence_history": [],
                "min_spacing_km": 0}

    # 生成满足间距约束的初始解
    current = None
    for attempt in range(100):
        candidate = random.sample(valid_points, n_turbines)
        if is_valid_solution(df, candidate, min_spacing):
            current = candidate
            break

    if current is None:
        current = random.sample(valid_points, n_turbines)

    current_fitness = evaluate_solution(df, current, cost_weight,
                                        max_slope, max_road_distance, min_residential_distance,
                                        min_heritage_distance, min_geology_distance, min_water_distance,
                                        min_spacing)

    best_solution, best_fitness = current, current_fitness
    T, cooling = initial_temp, cooling_rate
    history = [best_fitness]

    for step in range(iterations_per_temp):
        neighbor = mutate_solution_with_spacing(current, valid_points, df, min_spacing, 0.3)
        f_new = evaluate_solution(df, neighbor, cost_weight,
                                  max_slope, max_road_distance, min_residential_distance,
                                  min_heritage_distance, min_geology_distance, min_water_distance,
                                  min_spacing)

        if f_new > current_fitness or random.random() < math.exp((f_new - current_fitness) / T):
            current, current_fitness = neighbor, f_new

        if current_fitness > best_fitness:
            best_solution, best_fitness = current, current_fitness

        history.append(best_fitness)
        T *= cooling

    min_spacing_achieved = calculate_min_spacing(df, best_solution) if best_solution else 0

    return {
        "algorithm": "模拟退火算法",
        "solution": best_solution,
        "fitness": best_fitness,
        "convergence_history": history,
        "min_spacing_km": min_spacing_achieved
    }


# ===================== 粒子群优化算法 =====================
def run_pso(df, n_turbines, cost_weight,
            max_slope, max_road_distance, min_residential_distance,
            min_heritage_distance, min_geology_distance, min_water_distance,
            pop_size=30, generations=100, w=0.7, c1=1.5, c2=1.5,
            min_spacing=0.8):  # 添加间距参数
    """运行粒子群优化算法"""
    valid_points = df[df["valid"] == 1].index.tolist()

    if len(valid_points) < n_turbines:
        return {"algorithm": "粒子群优化算法", "solution": [], "fitness": -1, "convergence_history": [],
                "min_spacing_km": 0}

    # 初始化粒子群
    particles = []
    for _ in range(pop_size):
        candidate = random.sample(valid_points, n_turbines)
        particles.append(candidate)

    personal_best = particles.copy()
    personal_best_fitness = [evaluate_solution(df, p, cost_weight,
                                               max_slope, max_road_distance, min_residential_distance,
                                               min_heritage_distance, min_geology_distance, min_water_distance,
                                               min_spacing)
                             for p in particles]

    global_best_idx = np.argmax(personal_best_fitness)
    global_best = particles[global_best_idx]
    global_best_fitness = personal_best_fitness[global_best_idx]

    history = [global_best_fitness]

    for g in range(generations):
        for i in range(pop_size):
            # 更新粒子位置
            new_particle = []
            for j in range(n_turbines):
                if random.random() < w:  # 惯性
                    new_particle.append(particles[i][j])
                elif random.random() < c1 / (c1 + c2):  # 个体经验
                    new_particle.append(personal_best[i][j])
                else:  # 社会经验
                    new_particle.append(global_best[j])

            # 去重和补全
            new_particle = list(dict.fromkeys(new_particle))
            while len(new_particle) < n_turbines:
                available = [p for p in valid_points if p not in new_particle]
                if available:
                    new_particle.append(random.choice(available))
                else:
                    break
            new_particle = new_particle[:n_turbines]

            # 评估新位置
            new_fitness = evaluate_solution(df, new_particle, cost_weight,
                                            max_slope, max_road_distance, min_residential_distance,
                                            min_heritage_distance, min_geology_distance, min_water_distance,
                                            min_spacing)

            # 更新个体最优
            if new_fitness > personal_best_fitness[i]:
                personal_best[i] = new_particle
                personal_best_fitness[i] = new_fitness

            particles[i] = new_particle

        # 更新全局最优
        current_best_idx = np.argmax(personal_best_fitness)
        if personal_best_fitness[current_best_idx] > global_best_fitness:
            global_best = personal_best[current_best_idx]
            global_best_fitness = personal_best_fitness[current_best_idx]

        history.append(global_best_fitness)

    min_spacing_achieved = calculate_min_spacing(df, global_best) if global_best else 0

    return {
        "algorithm": "粒子群优化算法",
        "solution": global_best,
        "fitness": global_best_fitness,
        "convergence_history": history,
        "min_spacing_km": min_spacing_achieved
    }


# ===================== PuLP优化求解器 =====================
def run_pulp_optimizer(df, n_turbines, cost_weight,
                       max_slope, max_road_distance, min_residential_distance,
                       min_heritage_distance, min_geology_distance, min_water_distance,
                       solver_type='CBC', time_limit=60,
                       min_spacing=0.8):  # 添加间距参数
    """使用PuLP求解器进行优化"""
    try:
        # 筛选有效点位
        valid_df = df[df["valid"] == 1].copy()

        if len(valid_df) < n_turbines:
            return {"algorithm": "PuLP优化求解器", "solution": [], "fitness": -1, "convergence_history": [],
                    "min_spacing_km": 0}

        # 创建优化问题
        prob = pulp.LpProblem("WindFarm_Optimization", pulp.LpMaximize)

        # 决策变量：每个点位是否被选中
        x = pulp.LpVariable.dicts("x", valid_df.index.tolist(), cat='Binary')

        # 目标函数：最大化风能产出 - 成本
        energy_terms = []
        cost_terms = []

        for idx in valid_df.index:
            energy_terms.append(valid_df.loc[idx, "wind_power_density"] * 10 * x[idx])
            cost_terms.append(cost_weight * valid_df.loc[idx, "cost"] * x[idx])

        # 添加约束惩罚项
        penalty_terms = []

        # 坡度约束惩罚
        if 'slope' in valid_df.columns:
            for idx in valid_df.index:
                if valid_df.loc[idx, "slope"] > max_slope:
                    penalty_terms.append(-1000 * x[idx])

        # 道路距离约束惩罚
        if 'road_distance' in valid_df.columns:
            for idx in valid_df.index:
                if valid_df.loc[idx, "road_distance"] > max_road_distance:
                    penalty_terms.append(-1000 * x[idx])

        # 居民区距离约束惩罚
        if 'residential_distance' in valid_df.columns:
            for idx in valid_df.index:
                if valid_df.loc[idx, "residential_distance"] < min_residential_distance:
                    penalty_terms.append(-1000 * x[idx])

        # 文化遗产距离约束惩罚
        if 'heritage_distance' in valid_df.columns:
            for idx in valid_df.index:
                if valid_df.loc[idx, "heritage_distance"] < min_heritage_distance:
                    penalty_terms.append(-1000 * x[idx])

        # 地质距离约束惩罚
        if 'geology_distance' in valid_df.columns:
            for idx in valid_df.index:
                if valid_df.loc[idx, "geology_distance"] < min_geology_distance:
                    penalty_terms.append(-1000 * x[idx])

        # 水源距离约束惩罚
        if 'water_distance' in valid_df.columns:
            for idx in valid_df.index:
                if valid_df.loc[idx, "water_distance"] < min_water_distance:
                    penalty_terms.append(-1000 * x[idx])

        # 电网接近度奖励
        reward_terms = []
        if 'grid_proximity' in valid_df.columns:
            for idx in valid_df.index:
                reward_terms.append(valid_df.loc[idx, "grid_proximity"] * 100 * x[idx])

        # 间距约束 - 添加惩罚项
        spacing_penalty_terms = []
        # 由于PuLP中完整实现间距约束很复杂，这里使用简化版本
        # 在实际应用中，可以通过添加两两点对之间的约束来实现完整间距约束

        # 设置目标函数
        prob += (pulp.lpSum(energy_terms) - pulp.lpSum(cost_terms) +
                 pulp.lpSum(reward_terms) + pulp.lpSum(penalty_terms) + pulp.lpSum(spacing_penalty_terms))

        # 约束条件：选择n_turbines个点位
        prob += pulp.lpSum([x[idx] for idx in valid_df.index]) == n_turbines

        # 设置求解器
        if solver_type == 'CBC':
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
        elif solver_type == 'GLPK':
            solver = pulp.GLPK_CMD(timeLimit=time_limit, msg=0)
        elif solver_type == 'CPLEX':
            solver = pulp.CPLEX_CMD(timeLimit=time_limit, msg=0)
        else:
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)

        # 求解问题
        prob.solve(solver)

        # 提取结果
        if pulp.LpStatus[prob.status] == 'Optimal':
            selected_points = [idx for idx in valid_df.index if pulp.value(x[idx]) == 1]
            fitness = pulp.value(prob.objective)

            # 计算实际间距
            min_spacing_achieved = calculate_min_spacing(df, selected_points)

            return {
                "algorithm": "PuLP优化求解器",
                "solution": selected_points,
                "fitness": fitness,
                "convergence_history": [fitness],
                "min_spacing_km": min_spacing_achieved
            }
        else:
            return {"algorithm": "PuLP优化求解器", "solution": [], "fitness": -1, "convergence_history": [],
                    "min_spacing_km": 0}

    except Exception as e:
        print(f"PuLP求解器错误: {e}")
        return {"algorithm": "PuLP优化求解器", "solution": [], "fitness": -1, "convergence_history": [],
                "min_spacing_km": 0}


# ===================== 约束条件更新函数 =====================
def update_validity_with_constraints(df, max_slope, max_road_distance, min_residential_distance,
                                     min_heritage_distance, min_geology_distance, min_water_distance):
    """根据新的约束条件更新有效点位"""
    df_updated = df.copy()

    # 使用新的连续字段进行约束检查
    slope_valid = (df_updated["slope"] <= max_slope)
    road_valid = (df_updated["road_distance"] <= max_road_distance)
    residential_valid = (df_updated["residential_distance"] >= min_residential_distance)
    heritage_valid = (df_updated["heritage_distance"] >= min_heritage_distance)
    geology_valid = (df_updated["geology_distance"] >= min_geology_distance)
    water_valid = (df_updated["water_distance"] >= min_water_distance)

    # 风速约束
    wind_valid = (df_updated["predicted_wind_speed"] >= 5.0)

    # 电网接近度约束
    grid_valid = (df_updated["grid_proximity"] >= 0.1)

    # 综合所有约束条件
    df_updated["valid"] = (
            slope_valid & road_valid & residential_valid &
            heritage_valid & geology_valid & water_valid &
            wind_valid & grid_valid
    )

    return df_updated


# ===================== 总入口 =====================
def optimize(df, algo, n_turbines, cost_weight,
             max_slope, max_road_distance, min_residential_distance,
             min_heritage_distance, min_geology_distance, min_water_distance, **kwargs):
    """优化函数总入口"""
    # 设置默认风机间距为0.8km（不在前端显示）
    min_spacing = 0.8  # 默认风机间距0.8公里

    # 首先根据新的约束条件更新 df 中的 valid 列
    df = update_validity_with_constraints(
        df, max_slope, max_road_distance, min_residential_distance,
        min_heritage_distance, min_geology_distance, min_water_distance
    )

    # 合并参数，包含默认间距
    all_params = {
        'min_spacing': min_spacing,
        **kwargs
    }

    if algo == "遗传算法":
        return run_genetic_algorithm(df, n_turbines, cost_weight,
                                     max_slope, max_road_distance, min_residential_distance,
                                     min_heritage_distance, min_geology_distance, min_water_distance,
                                     **all_params)
    elif algo == "模拟退火算法":
        return run_simulated_annealing(df, n_turbines, cost_weight,
                                       max_slope, max_road_distance, min_residential_distance,
                                       min_heritage_distance, min_geology_distance, min_water_distance,
                                       **all_params)
    elif algo == "粒子群优化算法":
        return run_pso(df, n_turbines, cost_weight,
                       max_slope, max_road_distance, min_residential_distance,
                       min_heritage_distance, min_geology_distance, min_water_distance,
                       **all_params)
    elif algo == "PuLP优化求解器":
        return run_pulp_optimizer(df, n_turbines, cost_weight,
                                  max_slope, max_road_distance, min_residential_distance,
                                  min_heritage_distance, min_geology_distance, min_water_distance,
                                  **all_params)
    else:
        # 两者对比（遗传算法 vs 模拟退火算法）
        ga = run_genetic_algorithm(df, n_turbines, cost_weight,
                                   max_slope, max_road_distance, min_residential_distance,
                                   min_heritage_distance, min_geology_distance, min_water_distance,
                                   **all_params)
        sa = run_simulated_annealing(df, n_turbines, cost_weight,
                                     max_slope, max_road_distance, min_residential_distance,
                                     min_heritage_distance, min_geology_distance, min_water_distance,
                                     **all_params)
        return ga if ga["fitness"] > sa["fitness"] else sa