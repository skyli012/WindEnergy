# src/optimization/algorithm_convergence_curve.py

import streamlit as st
import numpy as np
import pandas as pd
import pulp
import time
from scipy.optimize import minimize


def calculate_fitness(positions, df, cost_weight=0.5, **constraints):
    """基于真实数据计算适应度函数"""
    if len(positions) == 0:
        return 0

    # 获取选中的点位数据
    selected_data = df.loc[positions]

    # 1. 发电量收益（基于真实的风速数据）
    if 'predicted_wind_speed' in selected_data.columns:
        # 使用风功率公式: P = 0.5 * ρ * A * v³
        air_density = 1.225  # kg/m³
        rotor_diameter = 140  # 米
        rotor_area = np.pi * (rotor_diameter / 2) ** 2
        power_benefit = 0.5 * air_density * rotor_area * (selected_data['predicted_wind_speed'] ** 3).sum()
    else:
        power_benefit = 0

    # 2. 成本惩罚（基于真实的约束条件）
    cost_penalty = 0

    # 坡度约束惩罚
    if 'slope' in selected_data.columns:
        max_slope = constraints.get('max_slope', 15)
        slope_violation = selected_data[selected_data['slope'] > max_slope]['slope'].sum()
        cost_penalty += slope_violation * 10

    # 道路距离约束惩罚
    if 'road_distance' in selected_data.columns:
        max_road_distance = constraints.get('max_road_distance', 1000)
        road_violation = selected_data[selected_data['road_distance'] > max_road_distance]['road_distance'].sum()
        cost_penalty += road_violation * 0.1

    # 居民区距离约束惩罚
    if 'residential_distance' in selected_data.columns:
        min_residential_distance = constraints.get('min_residential_distance', 600)
        residential_violation = selected_data[selected_data['residential_distance'] < min_residential_distance]
        if len(residential_violation) > 0:
            violation_amount = (min_residential_distance - residential_violation['residential_distance']).sum()
            cost_penalty += violation_amount * 5

    # 文化遗产距离约束惩罚
    if 'heritage_distance' in selected_data.columns:
        min_heritage_distance = constraints.get('min_heritage_distance', 700)
        heritage_violation = selected_data[selected_data['heritage_distance'] < min_heritage_distance]
        if len(heritage_violation) > 0:
            violation_amount = (min_heritage_distance - heritage_violation['heritage_distance']).sum()
            cost_penalty += violation_amount * 8

    # 地质距离约束惩罚
    if 'geology_distance' in selected_data.columns:
        min_geology_distance = constraints.get('min_geology_distance', 800)
        geology_violation = selected_data[selected_data['geology_distance'] < min_geology_distance]
        if len(geology_violation) > 0:
            violation_amount = (min_geology_distance - geology_violation['geology_distance']).sum()
            cost_penalty += violation_amount * 6

    # 水体距离约束惩罚
    if 'water_distance' in selected_data.columns:
        min_water_distance = constraints.get('min_water_distance', 1000)
        water_violation = selected_data[selected_data['water_distance'] < min_water_distance]
        if len(water_violation) > 0:
            violation_amount = (min_water_distance - water_violation['water_distance']).sum()
            cost_penalty += violation_amount * 7

    # 建设成本
    if 'cost' in selected_data.columns:
        construction_cost = selected_data['cost'].sum() * 0.01
        cost_penalty += construction_cost

    # 最终适应度 = 发电量收益 - 成本惩罚
    fitness = power_benefit - cost_weight * cost_penalty

    return max(fitness, 0)  # 确保适应度非负


def calculate_real_power_generation(turbines_df):
    """基于真实风速数据计算发电量"""
    if turbines_df.empty:
        return None

    TURBINE_CONFIG = {
        'model': '金风科技 GW-140/2500',
        'rated_power': 2500,  # kW
        'rotor_diameter': 140,  # 米
        'hub_height': 90,  # 米
        'cut_in_speed': 3.0,  # m/s
        'rated_speed': 11.0,  # m/s
        'cut_out_speed': 25.0,  # m/s
        'efficiency': 0.45,  # 综合效率
    }

    def power_curve(wind_speed):
        """基于真实功率曲线计算输出功率"""
        if wind_speed < TURBINE_CONFIG['cut_in_speed']:
            return 0
        elif wind_speed < TURBINE_CONFIG['rated_speed']:
            # 立方关系计算功率
            return TURBINE_CONFIG['rated_power'] * (
                    (wind_speed ** 3 - TURBINE_CONFIG['cut_in_speed'] ** 3) /
                    (TURBINE_CONFIG['rated_speed'] ** 3 - TURBINE_CONFIG['cut_in_speed'] ** 3)
            )
        elif wind_speed <= TURBINE_CONFIG['cut_out_speed']:
            return TURBINE_CONFIG['rated_power']
        else:
            return 0

    annual_generation_per_turbine = []
    capacity_factors = []

    for _, turbine in turbines_df.iterrows():
        wind_speed = turbine.get('predicted_wind_speed', 0)

        # 计算理论功率输出
        theoretical_power = power_curve(wind_speed)

        # 考虑综合效率
        actual_power = theoretical_power * TURBINE_CONFIG['efficiency']

        # 年发电量 (kWh) - 8760小时/年
        annual_energy = actual_power * 8760

        annual_generation_per_turbine.append(annual_energy)

        # 容量因数
        capacity_factor = annual_energy / (TURBINE_CONFIG['rated_power'] * 8760)
        capacity_factors.append(capacity_factor)

    total_annual_generation = sum(annual_generation_per_turbine)
    avg_capacity_factor = np.mean(capacity_factors) if capacity_factors else 0
    total_capacity = len(turbines_df) * TURBINE_CONFIG['rated_power']
    equivalent_full_load_hours = total_annual_generation / total_capacity if total_capacity > 0 else 0

    # 计算真实的经济指标
    electricity_price = 0.4  # 元/kWh
    investment_per_kw = 6000  # 元/kW
    om_cost_per_kw = 150  # 元/kW/年

    total_investment = total_capacity * investment_per_kw
    annual_revenue = total_annual_generation * electricity_price
    annual_om_cost = total_capacity * om_cost_per_kw
    annual_profit = annual_revenue - annual_om_cost
    payback_period = total_investment / annual_profit if annual_profit > 0 else float('inf')

    return {
        'total_annual_generation_kwh': total_annual_generation,
        'total_annual_generation_mwh': total_annual_generation / 1000,
        'total_annual_generation_gwh': total_annual_generation / 1e6,
        'total_capacity_kw': total_capacity,
        'total_capacity_mw': total_capacity / 1000,
        'average_capacity_factor': avg_capacity_factor,
        'equivalent_full_load_hours': equivalent_full_load_hours,
        'annual_generation_per_turbine': annual_generation_per_turbine,
        'capacity_factors': capacity_factors,
        'turbine_config': TURBINE_CONFIG,
        'economic_analysis': {
            'total_investment': total_investment,
            'annual_revenue': annual_revenue,
            'annual_om_cost': annual_om_cost,
            'annual_profit': annual_profit,
            'payback_period': payback_period,
            'electricity_price': electricity_price,
            'investment_per_kw': investment_per_kw
        }
    }


def real_genetic_algorithm(df, n_turbines, pop_size=50, generations=100,
                           mutation_rate=0.1, crossover_rate=0.8, **kwargs):
    """真实的遗传算法实现"""
    start_time = time.time()

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    n_points = len(valid_points)
    fitness_history = []
    best_fitness_history = []

    # 初始化种群
    population = []
    for _ in range(pop_size):
        individual = np.random.choice(valid_points.index, n_turbines, replace=False)
        population.append(individual)

    best_fitness = -float('inf')
    best_individual = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    for generation in range(generations):
        # 计算适应度
        fitness_scores = []
        for individual in population:
            fitness = calculate_fitness(individual, df, **kwargs)
            fitness_scores.append(fitness)

        # 记录历史
        current_best_fitness = max(fitness_scores)
        best_fitness_history.append(current_best_fitness)
        avg_fitness = np.mean(fitness_scores)
        fitness_history.append(avg_fitness)

        # 更新全局最优
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[np.argmax(fitness_scores)].copy()

        # 选择（轮盘赌选择）
        fitness_scores = np.array(fitness_scores)
        if fitness_scores.min() < 0:
            fitness_scores = fitness_scores - fitness_scores.min() + 1e-6
        selection_probs = fitness_scores / fitness_scores.sum()

        new_population = []
        for _ in range(pop_size):
            parent_idx = np.random.choice(len(population), p=selection_probs)
            new_population.append(population[parent_idx].copy())

        # 交叉
        for i in range(0, len(new_population), 2):
            if i + 1 < len(new_population) and np.random.random() < crossover_rate:
                parent1 = new_population[i]
                parent2 = new_population[i + 1]

                # 单点交叉
                crossover_point = np.random.randint(1, n_turbines - 1)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

                # 确保不重复
                child1 = np.unique(child1)
                child2 = np.unique(child2)

                # 如果交叉后数量变化，随机补充
                while len(child1) < n_turbines:
                    new_gene = np.random.choice(valid_points.index)
                    if new_gene not in child1:
                        child1 = np.append(child1, new_gene)

                while len(child2) < n_turbines:
                    new_gene = np.random.choice(valid_points.index)
                    if new_gene not in child2:
                        child2 = np.append(child2, new_gene)

                new_population[i] = child1[:n_turbines]
                new_population[i + 1] = child2[:n_turbines]

        # 变异
        for i in range(len(new_population)):
            if np.random.random() < mutation_rate:
                individual = new_population[i]
                mutation_point = np.random.randint(n_turbines)
                new_gene = np.random.choice(valid_points.index)
                while new_gene in individual:
                    new_gene = np.random.choice(valid_points.index)
                individual[mutation_point] = new_gene

        population = new_population

        # 更新进度
        progress = (generation + 1) / generations
        progress_bar.progress(progress)
        status_text.text(f"遗传算法进度: {generation + 1}/{generations} 代, 当前最优适应度: {current_best_fitness:.2f}")

    progress_bar.empty()
    status_text.empty()

    computation_time = time.time() - start_time

    # 计算真实的最优位置数据
    best_positions_data = df.loc[best_individual] if best_individual is not None else pd.DataFrame()

    # 计算真实的发电量
    power_results = calculate_real_power_generation(best_positions_data)

    return {
        'best_positions': best_individual.tolist() if best_individual is not None else [],
        'best_positions_data': best_positions_data,
        'best_fitness': best_fitness,
        'fitness_history': best_fitness_history,
        'algorithm': '遗传算法',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs)
    }


def check_constraints_violations(positions_data, constraints):
    """检查约束违反情况"""
    violations = {}

    if 'slope' in positions_data.columns and 'max_slope' in constraints:
        slope_violations = positions_data[positions_data['slope'] > constraints['max_slope']]
        violations['slope'] = len(slope_violations)

    if 'road_distance' in positions_data.columns and 'max_road_distance' in constraints:
        road_violations = positions_data[positions_data['road_distance'] > constraints['max_road_distance']]
        violations['road'] = len(road_violations)

    # 添加其他约束检查...

    return violations


def real_simulated_annealing(df, n_turbines, **kwargs):
    """真实的模拟退火算法"""
    start_time = time.time()

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    # 初始解
    current_solution = np.random.choice(valid_points.index, n_turbines, replace=False)
    current_fitness = calculate_fitness(current_solution, df, **kwargs)

    best_solution = current_solution.copy()
    best_fitness = current_fitness

    initial_temp = kwargs.get('initial_temp', 1000)
    cooling_rate = kwargs.get('cooling_rate', 0.95)
    iterations_per_temp = kwargs.get('iterations_per_temp', 50)

    temperature = initial_temp
    fitness_history = [current_fitness]

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_iterations = int(np.log(0.01) / np.log(cooling_rate)) * iterations_per_temp
    current_iteration = 0

    while temperature > 1e-3:
        for _ in range(iterations_per_temp):
            # 生成邻域解
            neighbor = current_solution.copy()
            mutation_point = np.random.randint(n_turbines)
            new_gene = np.random.choice(valid_points.index)
            while new_gene in neighbor:
                new_gene = np.random.choice(valid_points.index)
            neighbor[mutation_point] = new_gene

            neighbor_fitness = calculate_fitness(neighbor, df, **kwargs)

            # 决定是否接受新解
            if neighbor_fitness > current_fitness:
                current_solution = neighbor
                current_fitness = neighbor_fitness
                if neighbor_fitness > best_fitness:
                    best_solution = neighbor.copy()
                    best_fitness = neighbor_fitness
            else:
                delta = neighbor_fitness - current_fitness
                acceptance_prob = np.exp(delta / temperature)
                if np.random.random() < acceptance_prob:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness

            fitness_history.append(current_fitness)
            current_iteration += 1

            # 更新进度
            if current_iteration % 10 == 0:
                progress = min(1.0, current_iteration / total_iterations)
                progress_bar.progress(progress)
                status_text.text(
                    f"模拟退火进度: {current_iteration}/{total_iterations}, 温度: {temperature:.2f}, 最优适应度: {best_fitness:.2f}")

        temperature *= cooling_rate

    progress_bar.empty()
    status_text.empty()

    computation_time = time.time() - start_time
    best_positions_data = df.loc[best_solution]
    power_results = calculate_real_power_generation(best_positions_data)

    return {
        'best_positions': best_solution.tolist(),
        'best_positions_data': best_positions_data,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'algorithm': '模拟退火算法',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs)
    }


def real_particle_swarm(df, n_turbines, pop_size=30, generations=100,
                        w=0.7, c1=1.5, c2=1.5, **kwargs):
    """真实的粒子群优化算法"""
    start_time = time.time()

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    n_points = len(valid_points)

    # 初始化粒子群
    particles = []
    velocities = []
    personal_best_positions = []
    personal_best_fitnesses = []

    for _ in range(pop_size):
        # 随机选择风机位置
        positions = np.random.choice(valid_points.index, n_turbines, replace=False)
        particles.append(positions)
        velocities.append(np.zeros(n_turbines))
        personal_best_positions.append(positions.copy())
        fitness = calculate_fitness(positions, df, **kwargs)
        personal_best_fitnesses.append(fitness)

    # 全局最优
    global_best_idx = np.argmax(personal_best_fitnesses)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_fitness = personal_best_fitnesses[global_best_idx]

    fitness_history = [global_best_fitness]

    progress_bar = st.progress(0)
    status_text = st.empty()

    for generation in range(generations):
        for i in range(pop_size):
            # 更新粒子位置
            for d in range(n_turbines):
                # PSO速度更新公式
                r1, r2 = np.random.random(), np.random.random()
                velocities[i][d] = (w * velocities[i][d] +
                                    c1 * r1 * (personal_best_positions[i][d] - particles[i][d]) +
                                    c2 * r2 * (global_best_position[d] - particles[i][d]))

                # 位置更新 - 转换为离散索引
                new_index = int(particles[i][d] + velocities[i][d])
                new_index = max(0, min(new_index, n_points - 1))

                # 确保不重复
                if new_index not in particles[i]:
                    particles[i][d] = new_index

            # 计算适应度
            current_fitness = calculate_fitness(particles[i], df, **kwargs)

            # 更新个体最优
            if current_fitness > personal_best_fitnesses[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_fitnesses[i] = current_fitness

                # 更新全局最优
                if current_fitness > global_best_fitness:
                    global_best_position = particles[i].copy()
                    global_best_fitness = current_fitness

        fitness_history.append(global_best_fitness)

        # 更新进度
        progress = (generation + 1) / generations
        progress_bar.progress(progress)
        status_text.text(f"粒子群进度: {generation + 1}/{generations}, 最优适应度: {global_best_fitness:.2f}")

    progress_bar.empty()
    status_text.empty()

    computation_time = time.time() - start_time
    best_positions_data = df.loc[global_best_position]
    power_results = calculate_real_power_generation(best_positions_data)

    return {
        'best_positions': global_best_position.tolist(),
        'best_positions_data': best_positions_data,
        'best_fitness': global_best_fitness,
        'fitness_history': fitness_history,
        'algorithm': '粒子群优化算法',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs)
    }


def real_pulp_optimization(df, n_turbines, solver_type="CBC", time_limit=60, **kwargs):
    """使用PuLP进行数学规划优化"""
    start_time = time.time()

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    # 创建问题
    prob = pulp.LpProblem("WindFarm_Optimization", pulp.LpMaximize)

    # 决策变量：是否选择该点位
    x = pulp.LpVariable.dicts("x", valid_points.index, cat='Binary')

    # 目标函数：最大化发电量，最小化成本
    power_terms = []
    cost_terms = []

    for idx, point in valid_points.iterrows():
        # 发电量项
        wind_speed = point.get('predicted_wind_speed', 0)
        power_value = 0.5 * 1.225 * (wind_speed ** 3) * 1000  # 简化计算
        power_terms.append(power_value * x[idx])

        # 成本项
        cost_value = 0
        if point.get('slope', 0) > kwargs.get('max_slope', 15):
            cost_value += point['slope'] * 10
        cost_terms.append(cost_value * x[idx])

    # 目标函数
    cost_weight = kwargs.get('cost_weight', 0.5)
    prob += pulp.lpSum(power_terms) - cost_weight * pulp.lpSum(cost_terms)

    # 约束：选择恰好n_turbines个点位
    prob += pulp.lpSum([x[i] for i in valid_points.index]) == n_turbines

    # 求解
    if solver_type == "CBC":
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit)
    elif solver_type == "GLPK":
        solver = pulp.GLPK_CMD(timeLimit=time_limit)
    else:
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit)

    prob.solve(solver)

    # 提取结果
    selected_positions = []
    for idx in valid_points.index:
        if pulp.value(x[idx]) == 1:
            selected_positions.append(idx)

    best_fitness = pulp.value(prob.objective)
    computation_time = time.time() - start_time

    best_positions_data = df.loc[selected_positions]
    power_results = calculate_real_power_generation(best_positions_data)

    return {
        'best_positions': selected_positions,
        'best_positions_data': best_positions_data,
        'best_fitness': best_fitness if best_fitness else 0,
        'fitness_history': [best_fitness] if best_fitness else [0],
        'algorithm': 'PuLP优化求解器',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs)
    }


def call_optimize_function(df, algo, algorithm_params):
    """调用真实优化函数"""

    # 参数映射和转换
    optimization_params = algorithm_params.copy()

    # 处理风场数量相关的参数
    if 'total_turbines' in optimization_params:
        # 多风场优化：使用总风机数量
        optimization_params['n_turbines'] = optimization_params['total_turbines']
    elif 'n_turbines_per_farm' in optimization_params:
        # 单风场优化：使用单场风机数量
        optimization_params['n_turbines'] = optimization_params['n_turbines_per_farm']

    # 移除可能冲突的参数
    optimization_params.pop('n_farms', None)
    optimization_params.pop('n_turbines_per_farm', None)
    optimization_params.pop('total_turbines', None)
    optimization_params.pop('min_farm_distance', None)

    try:
        if algo == "遗传算法":
            result = real_genetic_algorithm(df, **optimization_params)
        elif algo == "模拟退火算法":
            result = real_simulated_annealing(df, **optimization_params)
        elif algo == "粒子群优化算法":
            result = real_particle_swarm(df, **optimization_params)
        elif algo == "PuLP优化求解器":
            result = real_pulp_optimization(df, **optimization_params)
        else:
            result = real_genetic_algorithm(df, **optimization_params)

        return result

    except Exception as e:
        st.error(f"优化算法执行错误: {str(e)}")
        # 回退到基础遗传算法
        st.info("尝试使用基础参数重新计算...")
        base_params = {
            'n_turbines': optimization_params.get('n_turbines', 5),
            'pop_size': 30,
            'generations': 50
        }
        return real_genetic_algorithm(df, **base_params)