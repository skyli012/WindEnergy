# src/optimization/algorithm_convergence_curve.py

import streamlit as st
import numpy as np
import pandas as pd
import pulp
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


# ==============================
# 储能经济性计算函数
# ==============================

def calculate_storage_investment_cost(capacity_kwh, power_kw):
    """计算储能投资成本"""
    # 成本参数（元/kWh 和 元/kW）
    capacity_cost_per_kwh = 1500  # 元/kWh
    power_cost_per_kw = 1000  # 元/kW

    total_cost = capacity_kwh * capacity_cost_per_kwh + power_kw * power_cost_per_kw
    return total_cost


def calculate_storage_annual_revenue(selected_data, capacity_kwh, power_kw, constraints):
    """计算储能年收益"""
    strategy = constraints.get('storage_strategy', '平滑输出')
    electricity_price = 0.4  # 元/kWh

    if strategy == '经济调度' or strategy == '削峰填谷':
        # 峰谷套利收益
        peak_valley_diff = constraints.get('peak_valley_diff', 0.7)  # 峰谷电价差
        daily_cycles = 1.0  # 每天充放电次数
        efficiency = 0.85  # 充放电效率

        # 考虑容量和功率限制
        usable_capacity = capacity_kwh * 0.8  # 可用容量（考虑SOC范围）
        daily_revenue = min(usable_capacity * peak_valley_diff * efficiency * daily_cycles,
                            power_kw * 24 * peak_valley_diff * 0.5)  # 功率限制
        annual_revenue = daily_revenue * 365
    else:
        # 其他策略收益（平滑、削峰等）
        capacity_utilization = 0.3  # 假设30%的容量利用率
        annual_revenue = capacity_kwh * electricity_price * capacity_utilization * 365

    return annual_revenue


def calculate_storage_operation_cost(capacity_kwh, power_kw):
    """计算储能年运行维护成本"""
    # O&M成本（元/kWh/年）
    om_cost_per_kwh = 50
    return capacity_kwh * om_cost_per_kwh


def calculate_wind_utilization(wind_speed_series):
    """
    计算风能利用率指标
    基于风速的稳定性、可利用小时数等因素
    """
    if len(wind_speed_series) == 0:
        return 0

    # 风速在风机工作范围内的比例（3-25 m/s）
    operational_hours = ((wind_speed_series >= 3) & (wind_speed_series <= 25)).mean()

    # 风速稳定性（标准差越小越稳定）
    wind_std = wind_speed_series.std()
    stability = 1 / (1 + wind_std)  # 标准化稳定性指标

    # 高风速利用率（>7 m/s 的比例）
    high_wind_hours = (wind_speed_series >= 7).mean()

    # 综合利用率指标
    utilization_rate = 0.5 * operational_hours + 0.3 * stability + 0.2 * high_wind_hours

    return utilization_rate


def calculate_composite_fitness_with_storage(positions, df, wind_speed_weight=0.6, utilization_weight=0.4,
                                             **constraints):
    """基于风速、风能利用率和储能经济性的综合适应度函数"""
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

        # 计算风速得分（归一化）
        wind_speeds = selected_data['predicted_wind_speed']
        max_wind_speed = df['predicted_wind_speed'].max()
        normalized_wind_speed = wind_speeds.sum() / (len(wind_speeds) * max_wind_speed) if max_wind_speed > 0 else 0

        # 计算风能利用率得分
        if 'wind_utilization_rate' in selected_data.columns:
            utilization_scores = selected_data['wind_utilization_rate']
            max_utilization = df['wind_utilization_rate'].max()
            normalized_utilization = utilization_scores.sum() / (
                    len(utilization_scores) * max_utilization) if max_utilization > 0 else 0
        else:
            # 如果没有预计算的利用率，实时计算
            utilization_rates = []
            for idx in positions:
                point_data = df.loc[idx]
                # 这里简化计算，实际应该使用时间序列数据
                utilization = calculate_wind_utilization(pd.Series([point_data.get('predicted_wind_speed', 0)]))
                utilization_rates.append(utilization)
            normalized_utilization = sum(utilization_rates) / len(utilization_rates) if utilization_rates else 0

        # 综合评分
        composite_score = (wind_speed_weight * normalized_wind_speed +
                           utilization_weight * normalized_utilization)

        # 基础发电量计算（用于约束惩罚的基准）
        power_benefit = 0.5 * air_density * rotor_area * (wind_speeds ** 3).sum()
    else:
        composite_score = 0
        power_benefit = 0

    # 2. 储能经济性计算
    storage_economic_score = 0
    if 'storage_capacity' in constraints and 'storage_power' in constraints:
        storage_capacity_kwh = constraints.get('storage_capacity', 0)
        storage_power_kw = constraints.get('storage_power', 0)

        if storage_capacity_kwh > 0 and storage_power_kw > 0:
            # 计算储能投资成本
            storage_investment = calculate_storage_investment_cost(storage_capacity_kwh, storage_power_kw)

            # 计算储能年收益
            storage_annual_revenue = calculate_storage_annual_revenue(
                selected_data, storage_capacity_kwh, storage_power_kw, constraints
            )

            # 计算储能运维成本
            storage_om_cost = calculate_storage_operation_cost(storage_capacity_kwh, storage_power_kw)

            # 计算储能净年收益
            storage_net_annual_benefit = storage_annual_revenue - storage_om_cost

            # 计算储能投资回收期（简化）
            storage_payback_years = storage_investment / storage_net_annual_benefit if storage_net_annual_benefit > 0 else float(
                'inf')

            # 储能经济性评分（回收期越短评分越高）
            if storage_payback_years < 5:
                storage_economic_score = 1.0
            elif storage_payback_years < 10:
                storage_economic_score = 0.8
            elif storage_payback_years < 15:
                storage_economic_score = 0.5
            elif storage_payback_years < 20:
                storage_economic_score = 0.3
            else:
                storage_economic_score = 0.1

            # 添加到适应度中（适当权重）
            storage_weight = constraints.get('storage_weight', 0.3)  # 储能经济性权重
            composite_score += storage_economic_score * storage_weight

    # 3. 成本惩罚（基于真实的约束条件）
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

    # 风场间距约束惩罚
    if 'min_farm_distance' in constraints and len(positions) > 1:
        coords = selected_data[['lat', 'lon']].values
        if len(coords) > 1:
            distances = pdist(coords) * 111000  # 转换为米（近似）
            min_distance = distances.min() if len(distances) > 0 else 0
            min_required = constraints['min_farm_distance']
            if min_distance < min_required:
                cost_penalty += (min_required - min_distance) * 100

    cost_weight = constraints.get('cost_weight', 0.5)

    # 最终适应度 = 综合评分 - 成本惩罚
    fitness = composite_score * 1000 - cost_weight * cost_penalty  # 缩放综合评分

    return max(fitness, 0)  # 确保适应度非负


def calculate_composite_fitness(positions, df, wind_speed_weight=0.6, utilization_weight=0.4, **constraints):
    """兼容旧版本的适应度函数，可选择是否包含储能优化"""
    if constraints.get('enable_storage_optimization', False):
        return calculate_composite_fitness_with_storage(positions, df, wind_speed_weight, utilization_weight,
                                                        **constraints)
    else:
        # 原版的适应度计算（不含储能优化）
        if len(positions) == 0:
            return 0

        selected_data = df.loc[positions]

        if 'predicted_wind_speed' in selected_data.columns:
            wind_speeds = selected_data['predicted_wind_speed']
            max_wind_speed = df['predicted_wind_speed'].max()
            normalized_wind_speed = wind_speeds.sum() / (len(wind_speeds) * max_wind_speed) if max_wind_speed > 0 else 0

            if 'wind_utilization_rate' in selected_data.columns:
                utilization_scores = selected_data['wind_utilization_rate']
                max_utilization = df['wind_utilization_rate'].max()
                normalized_utilization = utilization_scores.sum() / (
                        len(utilization_scores) * max_utilization) if max_utilization > 0 else 0
            else:
                normalized_utilization = 0

            composite_score = (wind_speed_weight * normalized_wind_speed +
                               utilization_weight * normalized_utilization)
        else:
            composite_score = 0

        cost_penalty = 0
        if 'slope' in selected_data.columns:
            max_slope = constraints.get('max_slope', 15)
            slope_violation = selected_data[selected_data['slope'] > max_slope]['slope'].sum()
            cost_penalty += slope_violation * 10

        cost_weight = constraints.get('cost_weight', 0.5)
        fitness = composite_score * 1000 - cost_weight * cost_penalty

        return max(fitness, 0)


def calculate_fitness(positions, df, cost_weight=0.5, **constraints):
    """兼容旧版本的适应度函数，默认使用综合评分"""
    return calculate_composite_fitness(positions, df, cost_weight=cost_weight, **constraints)


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
    utilization_rates = []

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

        # 风能利用率
        if 'wind_utilization_rate' in turbine:
            utilization_rates.append(turbine['wind_utilization_rate'])
        else:
            # 简化计算利用率
            utilization = 1.0 if 3 <= wind_speed <= 25 else 0.5
            utilization_rates.append(utilization)

    total_annual_generation = sum(annual_generation_per_turbine)
    avg_capacity_factor = np.mean(capacity_factors) if capacity_factors else 0
    avg_utilization_rate = np.mean(utilization_rates) if utilization_rates else 0
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
        'average_utilization_rate': avg_utilization_rate,
        'equivalent_full_load_hours': equivalent_full_load_hours,
        'annual_generation_per_turbine': annual_generation_per_turbine,
        'capacity_factors': capacity_factors,
        'utilization_rates': utilization_rates,
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


def real_genetic_algorithm_with_storage(df, n_turbines, pop_size=50, generations=100,
                                        mutation_rate=0.1, crossover_rate=0.8, **kwargs):
    """包含储能优化的遗传算法"""
    start_time = time.time()

    # 是否启用储能优化
    enable_storage_opt = kwargs.get('enable_storage_optimization', False)

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    n_points = len(valid_points)
    fitness_history = []
    best_fitness_history = []

    if enable_storage_opt:
        # 扩展个体编码：包含储能容量和功率
        # 前n_turbines个基因是风机位置，后2个基因是储能容量和功率
        individual_length = n_turbines + 2
    else:
        individual_length = n_turbines

    # 初始化种群
    population = []
    for _ in range(pop_size):
        if enable_storage_opt:
            # 风机位置（离散）
            turbine_genes = np.random.choice(valid_points.index, n_turbines, replace=False)
            # 储能容量和功率（连续）
            storage_capacity = np.random.uniform(kwargs.get('min_storage_capacity', 10000),
                                                 kwargs.get('max_storage_capacity', 200000))
            storage_power = np.random.uniform(kwargs.get('min_storage_power', 5000),
                                              kwargs.get('max_storage_power', 100000))
            individual = np.concatenate([turbine_genes, [storage_capacity, storage_power]])
        else:
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
            if enable_storage_opt:
                turbine_positions = individual[:n_turbines].astype(int)
                storage_capacity = individual[n_turbines]
                storage_power = individual[n_turbines + 1]

                # 添加储能参数到约束中
                current_constraints = kwargs.copy()
                current_constraints['storage_capacity'] = storage_capacity
                current_constraints['storage_power'] = storage_power
                current_constraints['enable_storage_optimization'] = True

                fitness = calculate_composite_fitness_with_storage(
                    turbine_positions, df, **current_constraints
                )
            else:
                fitness = calculate_composite_fitness(individual, df, **kwargs)
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

                # 对风机位置进行交叉
                if enable_storage_opt:
                    crossover_point = np.random.randint(1, n_turbines - 1)
                    child1_genes = np.concatenate([parent1[:crossover_point], parent2[crossover_point:n_turbines]])
                    child2_genes = np.concatenate([parent2[:crossover_point], parent1[crossover_point:n_turbines]])

                    # 对储能参数进行算术交叉
                    alpha = np.random.random()
                    storage1 = alpha * parent1[n_turbines:] + (1 - alpha) * parent2[n_turbines:]
                    storage2 = alpha * parent2[n_turbines:] + (1 - alpha) * parent1[n_turbines:]

                    child1 = np.concatenate([child1_genes, storage1])
                    child2 = np.concatenate([child2_genes, storage2])
                else:
                    crossover_point = np.random.randint(1, n_turbines - 1)
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

                # 确保风机位置不重复
                if enable_storage_opt:
                    turbine_genes1 = child1[:n_turbines].astype(int)
                    turbine_genes2 = child2[:n_turbines].astype(int)

                    # 去重并补充
                    unique1 = np.unique(turbine_genes1)
                    while len(unique1) < n_turbines:
                        new_gene = np.random.choice(valid_points.index)
                        if new_gene not in unique1:
                            unique1 = np.append(unique1, new_gene)

                    unique2 = np.unique(turbine_genes2)
                    while len(unique2) < n_turbines:
                        new_gene = np.random.choice(valid_points.index)
                        if new_gene not in unique2:
                            unique2 = np.append(unique2, new_gene)

                    child1[:n_turbines] = unique1[:n_turbines]
                    child2[:n_turbines] = unique2[:n_turbines]
                else:
                    child1 = np.unique(child1)
                    child2 = np.unique(child2)

                    while len(child1) < n_turbines:
                        new_gene = np.random.choice(valid_points.index)
                        if new_gene not in child1:
                            child1 = np.append(child1, new_gene)

                    while len(child2) < n_turbines:
                        new_gene = np.random.choice(valid_points.index)
                        if new_gene not in child2:
                            child2 = np.append(child2, new_gene)

                new_population[i] = child1[:individual_length]
                new_population[i + 1] = child2[:individual_length]

        # 变异
        for i in range(len(new_population)):
            if np.random.random() < mutation_rate:
                individual = new_population[i]
                if enable_storage_opt:
                    # 随机选择变异类型：风机位置变异或储能参数变异
                    if np.random.random() < 0.7:  # 70%概率变异风机位置
                        mutation_point = np.random.randint(n_turbines)
                        new_gene = np.random.choice(valid_points.index)
                        while new_gene in individual[:n_turbines]:
                            new_gene = np.random.choice(valid_points.index)
                        individual[mutation_point] = new_gene
                    else:  # 30%概率变异储能参数
                        mutation_point = n_turbines + np.random.randint(2)
                        if mutation_point == n_turbines:  # 变异容量
                            min_cap = kwargs.get('min_storage_capacity', 10000)
                            max_cap = kwargs.get('max_storage_capacity', 200000)
                            individual[mutation_point] = np.random.uniform(min_cap, max_cap)
                        else:  # 变异功率
                            min_pow = kwargs.get('min_storage_power', 5000)
                            max_pow = kwargs.get('max_storage_power', 100000)
                            individual[mutation_point] = np.random.uniform(min_pow, max_pow)
                else:
                    mutation_point = np.random.randint(n_turbines)
                    new_gene = np.random.choice(valid_points.index)
                    while new_gene in individual:
                        new_gene = np.random.choice(valid_points.index)
                    individual[mutation_point] = new_gene

        population = new_population

        # 更新进度
        progress = (generation + 1) / generations
        progress_bar.progress(progress)
        if enable_storage_opt:
            status_text.text(
                f"储能优化遗传算法进度: {generation + 1}/{generations} 代, 当前最优适应度: {current_best_fitness:.2f}")
        else:
            status_text.text(
                f"遗传算法进度: {generation + 1}/{generations} 代, 当前最优适应度: {current_best_fitness:.2f}")

    progress_bar.empty()
    status_text.empty()

    computation_time = time.time() - start_time

    # 提取最优解
    if enable_storage_opt and best_individual is not None:
        best_turbine_positions = best_individual[:n_turbines].astype(int).tolist()
        best_storage_capacity = best_individual[n_turbines]
        best_storage_power = best_individual[n_turbines + 1]
    else:
        best_turbine_positions = best_individual.tolist() if best_individual is not None else []
        best_storage_capacity = kwargs.get('storage_capacity', 0)
        best_storage_power = kwargs.get('storage_power', 0)

    # 计算真实的最优位置数据
    best_positions_data = df.loc[best_turbine_positions] if len(best_turbine_positions) > 0 else pd.DataFrame()

    # 计算真实的发电量
    power_results = calculate_real_power_generation(best_positions_data)

    # 计算储能经济性
    storage_economic_analysis = {}
    if enable_storage_opt:
        storage_investment = calculate_storage_investment_cost(best_storage_capacity, best_storage_power)
        storage_annual_revenue = calculate_storage_annual_revenue(
            best_positions_data, best_storage_capacity, best_storage_power, kwargs
        )
        storage_om_cost = calculate_storage_operation_cost(best_storage_capacity, best_storage_power)
        storage_net_benefit = storage_annual_revenue - storage_om_cost
        storage_payback = storage_investment / storage_net_benefit if storage_net_benefit > 0 else float('inf')

        storage_economic_analysis = {
            'storage_capacity_kwh': best_storage_capacity,
            'storage_power_kw': best_storage_power,
            'storage_investment': storage_investment,
            'storage_annual_revenue': storage_annual_revenue,
            'storage_om_cost': storage_om_cost,
            'storage_net_benefit': storage_net_benefit,
            'storage_payback_years': storage_payback
        }

    # 添加权重信息到结果中
    result = {
        'best_positions': best_turbine_positions,
        'best_positions_data': best_positions_data,
        'best_fitness': best_fitness,
        'fitness_history': best_fitness_history,
        'algorithm': '遗传算法（含储能优化）' if enable_storage_opt else '遗传算法',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs),
        'optimization_weights': {
            'wind_speed_weight': kwargs.get('wind_speed_weight', 0.6),
            'utilization_weight': kwargs.get('utilization_weight', 0.4),
            'storage_weight': kwargs.get('storage_weight', 0.3) if enable_storage_opt else 0
        },
        'n_farms': kwargs.get('n_farms', 1),
        'n_turbines_per_farm': n_turbines // kwargs.get('n_farms', 1),
        'enable_storage_optimization': enable_storage_opt,
        'storage_economic_analysis': storage_economic_analysis
    }

    return result


def real_genetic_algorithm(df, n_turbines, pop_size=50, generations=100,
                           mutation_rate=0.1, crossover_rate=0.8, **kwargs):
    """原始的遗传算法实现 - 调用新版本的函数但不启用储能优化"""
    kwargs['enable_storage_optimization'] = False
    return real_genetic_algorithm_with_storage(df, n_turbines, pop_size, generations,
                                               mutation_rate, crossover_rate, **kwargs)


def real_simulated_annealing(df, n_turbines, **kwargs):
    """真实的模拟退火算法 - 使用综合适应度函数"""
    start_time = time.time()

    enable_storage_opt = kwargs.get('enable_storage_optimization', False)

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    # 初始解
    if enable_storage_opt:
        # 初始解包含储能参数
        current_turbine_solution = np.random.choice(valid_points.index, n_turbines, replace=False)
        current_storage_capacity = np.random.uniform(kwargs.get('min_storage_capacity', 10000),
                                                     kwargs.get('max_storage_capacity', 200000))
        current_storage_power = np.random.uniform(kwargs.get('min_storage_power', 5000),
                                                  kwargs.get('max_storage_power', 100000))
        current_solution = (current_turbine_solution, current_storage_capacity, current_storage_power)
    else:
        current_solution = np.random.choice(valid_points.index, n_turbines, replace=False)

    # 计算初始适应度
    if enable_storage_opt:
        current_constraints = kwargs.copy()
        current_constraints['storage_capacity'] = current_storage_capacity
        current_constraints['storage_power'] = current_storage_power
        current_constraints['enable_storage_optimization'] = True
        current_fitness = calculate_composite_fitness_with_storage(
            current_turbine_solution, df, **current_constraints
        )
    else:
        current_fitness = calculate_composite_fitness(current_solution, df, **kwargs)

    best_solution = current_solution
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
            if enable_storage_opt:
                # 生成邻域解
                current_turbines, current_capacity, current_power = current_solution
                neighbor_turbines = current_turbines.copy()

                # 随机决定变异类型
                if np.random.random() < 0.7:  # 70%概率变异风机位置
                    mutation_point = np.random.randint(n_turbines)
                    new_gene = np.random.choice(valid_points.index)
                    while new_gene in neighbor_turbines:
                        new_gene = np.random.choice(valid_points.index)
                    neighbor_turbines[mutation_point] = new_gene
                    neighbor_capacity = current_capacity
                    neighbor_power = current_power
                else:  # 30%概率变异储能参数
                    neighbor_turbines = current_turbines.copy()
                    if np.random.random() < 0.5:  # 变异容量
                        neighbor_capacity = current_capacity + np.random.normal(0, current_capacity * 0.1)
                        neighbor_capacity = max(kwargs.get('min_storage_capacity', 10000),
                                                min(kwargs.get('max_storage_capacity', 200000), neighbor_capacity))
                        neighbor_power = current_power
                    else:  # 变异功率
                        neighbor_power = current_power + np.random.normal(0, current_power * 0.1)
                        neighbor_power = max(kwargs.get('min_storage_power', 5000),
                                             min(kwargs.get('max_storage_power', 100000), neighbor_power))
                        neighbor_capacity = current_capacity

                neighbor_solution = (neighbor_turbines, neighbor_capacity, neighbor_power)

                # 计算邻域解适应度
                neighbor_constraints = kwargs.copy()
                neighbor_constraints['storage_capacity'] = neighbor_capacity
                neighbor_constraints['storage_power'] = neighbor_power
                neighbor_constraints['enable_storage_optimization'] = True
                neighbor_fitness = calculate_composite_fitness_with_storage(
                    neighbor_turbines, df, **neighbor_constraints
                )
            else:
                # 生成邻域解
                neighbor = current_solution.copy()
                mutation_point = np.random.randint(n_turbines)
                new_gene = np.random.choice(valid_points.index)
                while new_gene in neighbor:
                    new_gene = np.random.choice(valid_points.index)
                neighbor[mutation_point] = new_gene

                neighbor_fitness = calculate_composite_fitness(neighbor, df, **kwargs)
                neighbor_solution = neighbor

            # 决定是否接受新解
            if neighbor_fitness > current_fitness:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                if neighbor_fitness > best_fitness:
                    best_solution = neighbor_solution
                    best_fitness = neighbor_fitness
            else:
                delta = neighbor_fitness - current_fitness
                acceptance_prob = np.exp(delta / temperature)
                if np.random.random() < acceptance_prob:
                    current_solution = neighbor_solution
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

    # 提取最优解
    if enable_storage_opt:
        best_turbines, best_capacity, best_power = best_solution
        best_positions_data = df.loc[best_turbines]

        # 计算储能经济性
        storage_investment = calculate_storage_investment_cost(best_capacity, best_power)
        storage_annual_revenue = calculate_storage_annual_revenue(
            best_positions_data, best_capacity, best_power, kwargs
        )
        storage_om_cost = calculate_storage_operation_cost(best_capacity, best_power)
        storage_net_benefit = storage_annual_revenue - storage_om_cost
        storage_payback = storage_investment / storage_net_benefit if storage_net_benefit > 0 else float('inf')

        storage_economic_analysis = {
            'storage_capacity_kwh': best_capacity,
            'storage_power_kw': best_power,
            'storage_investment': storage_investment,
            'storage_annual_revenue': storage_annual_revenue,
            'storage_om_cost': storage_om_cost,
            'storage_net_benefit': storage_net_benefit,
            'storage_payback_years': storage_payback
        }
    else:
        best_turbines = best_solution
        best_positions_data = df.loc[best_turbines]
        storage_economic_analysis = {}

    power_results = calculate_real_power_generation(best_positions_data)

    return {
        'best_positions': best_turbines.tolist(),
        'best_positions_data': best_positions_data,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'algorithm': '模拟退火算法（含储能优化）' if enable_storage_opt else '模拟退火算法',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs),
        'optimization_weights': {
            'wind_speed_weight': kwargs.get('wind_speed_weight', 0.6),
            'utilization_weight': kwargs.get('utilization_weight', 0.4),
            'storage_weight': kwargs.get('storage_weight', 0.3) if enable_storage_opt else 0
        },
        'enable_storage_optimization': enable_storage_opt,
        'storage_economic_analysis': storage_economic_analysis
    }


def real_particle_swarm(df, n_turbines, pop_size=30, generations=100,
                        w=0.7, c1=1.5, c2=1.5, **kwargs):
    """真实的粒子群优化算法 - 使用综合适应度函数"""
    start_time = time.time()

    enable_storage_opt = kwargs.get('enable_storage_optimization', False)

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    n_points = len(valid_points)

    if enable_storage_opt:
        # 扩展粒子维度：包含储能容量和功率
        dim = n_turbines + 2  # 风机位置 + 储能容量 + 储能功率
        # 定义边界
        bounds = []
        # 风机位置边界
        for _ in range(n_turbines):
            bounds.append([0, n_points - 1])
        # 储能容量边界
        bounds.append([kwargs.get('min_storage_capacity', 10000),
                       kwargs.get('max_storage_capacity', 200000)])
        # 储能功率边界
        bounds.append([kwargs.get('min_storage_power', 5000),
                       kwargs.get('max_storage_power', 100000)])
    else:
        dim = n_turbines
        bounds = [[0, n_points - 1] for _ in range(n_turbines)]

    # 初始化粒子群
    particles = []
    velocities = []
    personal_best_positions = []
    personal_best_fitnesses = []

    for _ in range(pop_size):
        # 初始化粒子位置
        if enable_storage_opt:
            position = []
            # 随机选择风机位置（离散）
            turbine_indices = np.random.choice(valid_points.index, n_turbines, replace=False)
            position.extend(turbine_indices)
            # 随机初始化储能参数
            position.append(np.random.uniform(bounds[n_turbines][0], bounds[n_turbines][1]))
            position.append(np.random.uniform(bounds[n_turbines + 1][0], bounds[n_turbines + 1][1]))
            position = np.array(position)
        else:
            position = np.random.choice(valid_points.index, n_turbines, replace=False)

        particles.append(position)
        velocities.append(np.zeros(dim))
        personal_best_positions.append(position.copy())

        # 计算适应度
        if enable_storage_opt:
            turbine_positions = position[:n_turbines].astype(int)
            storage_capacity = position[n_turbines]
            storage_power = position[n_turbines + 1]

            current_constraints = kwargs.copy()
            current_constraints['storage_capacity'] = storage_capacity
            current_constraints['storage_power'] = storage_power
            current_constraints['enable_storage_optimization'] = True

            fitness = calculate_composite_fitness_with_storage(
                turbine_positions, df, **current_constraints
            )
        else:
            fitness = calculate_composite_fitness(position, df, **kwargs)

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
            for d in range(dim):
                # PSO速度更新公式
                r1, r2 = np.random.random(), np.random.random()
                velocities[i][d] = (w * velocities[i][d] +
                                    c1 * r1 * (personal_best_positions[i][d] - particles[i][d]) +
                                    c2 * r2 * (global_best_position[d] - particles[i][d]))

                # 位置更新
                particles[i][d] = particles[i][d] + velocities[i][d]

                # 应用边界约束
                particles[i][d] = max(bounds[d][0], min(bounds[d][1], particles[i][d]))

            # 确保风机位置不重复（仅对前n_turbines维度）
            if enable_storage_opt:
                turbine_positions = particles[i][:n_turbines].copy()
                # 将连续值转换为离散索引
                discrete_indices = []
                for j in range(n_turbines):
                    idx = int(np.clip(turbine_positions[j], 0, n_points - 1))
                    discrete_indices.append(idx)

                # 去重处理
                unique_indices = np.unique(discrete_indices)
                while len(unique_indices) < n_turbines:
                    new_idx = np.random.randint(0, n_points)
                    if new_idx not in unique_indices:
                        unique_indices = np.append(unique_indices, new_idx)

                particles[i][:n_turbines] = unique_indices[:n_turbines]

            # 计算适应度
            if enable_storage_opt:
                turbine_positions = particles[i][:n_turbines].astype(int)
                storage_capacity = particles[i][n_turbines]
                storage_power = particles[i][n_turbines + 1]

                current_constraints = kwargs.copy()
                current_constraints['storage_capacity'] = storage_capacity
                current_constraints['storage_power'] = storage_power
                current_constraints['enable_storage_optimization'] = True

                current_fitness = calculate_composite_fitness_with_storage(
                    turbine_positions, df, **current_constraints
                )
            else:
                # 确保位置是整数
                int_positions = particles[i].astype(int)
                # 去重处理
                unique_positions = np.unique(int_positions)
                while len(unique_positions) < n_turbines:
                    new_idx = np.random.randint(0, n_points)
                    if new_idx not in unique_positions:
                        unique_positions = np.append(unique_positions, new_idx)

                particles[i] = unique_positions[:n_turbines]
                current_fitness = calculate_composite_fitness(particles[i], df, **kwargs)

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

    # 提取最优解
    if enable_storage_opt:
        best_turbine_positions = global_best_position[:n_turbines].astype(int).tolist()
        best_storage_capacity = global_best_position[n_turbines]
        best_storage_power = global_best_position[n_turbines + 1]
        best_positions_data = df.loc[best_turbine_positions]

        # 计算储能经济性
        storage_investment = calculate_storage_investment_cost(best_storage_capacity, best_storage_power)
        storage_annual_revenue = calculate_storage_annual_revenue(
            best_positions_data, best_storage_capacity, best_storage_power, kwargs
        )
        storage_om_cost = calculate_storage_operation_cost(best_storage_capacity, best_storage_power)
        storage_net_benefit = storage_annual_revenue - storage_om_cost
        storage_payback = storage_investment / storage_net_benefit if storage_net_benefit > 0 else float('inf')

        storage_economic_analysis = {
            'storage_capacity_kwh': best_storage_capacity,
            'storage_power_kw': best_storage_power,
            'storage_investment': storage_investment,
            'storage_annual_revenue': storage_annual_revenue,
            'storage_om_cost': storage_om_cost,
            'storage_net_benefit': storage_net_benefit,
            'storage_payback_years': storage_payback
        }
    else:
        best_turbine_positions = global_best_position.tolist()
        best_positions_data = df.loc[best_turbine_positions]
        storage_economic_analysis = {}

    power_results = calculate_real_power_generation(best_positions_data)

    return {
        'best_positions': best_turbine_positions,
        'best_positions_data': best_positions_data,
        'best_fitness': global_best_fitness,
        'fitness_history': fitness_history,
        'algorithm': '粒子群优化算法（含储能优化）' if enable_storage_opt else '粒子群优化算法',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs),
        'optimization_weights': {
            'wind_speed_weight': kwargs.get('wind_speed_weight', 0.6),
            'utilization_weight': kwargs.get('utilization_weight', 0.4),
            'storage_weight': kwargs.get('storage_weight', 0.3) if enable_storage_opt else 0
        },
        'enable_storage_optimization': enable_storage_opt,
        'storage_economic_analysis': storage_economic_analysis
    }


def real_pulp_optimization(df, n_turbines, solver_type="CBC", time_limit=60, **kwargs):
    """使用PuLP进行数学规划优化 - 使用综合评分"""
    start_time = time.time()

    enable_storage_opt = kwargs.get('enable_storage_optimization', False)

    valid_points = df[df['valid']] if 'valid' in df.columns else df
    if len(valid_points) < n_turbines:
        valid_points = df

    # 创建问题
    prob = pulp.LpProblem("WindFarm_Optimization", pulp.LpMaximize)

    # 决策变量：是否选择该点位
    x = pulp.LpVariable.dicts("x", valid_points.index, cat='Binary')

    # 目标函数：最大化综合评分
    composite_terms = []
    cost_terms = []

    wind_speed_weight = kwargs.get('wind_speed_weight', 0.6)
    utilization_weight = kwargs.get('utilization_weight', 0.4)

    # 预计算最大值为归一化
    max_wind_speed = valid_points['predicted_wind_speed'].max() if 'predicted_wind_speed' in valid_points.columns else 1
    max_utilization = valid_points[
        'wind_utilization_rate'].max() if 'wind_utilization_rate' in valid_points.columns else 1

    for idx, point in valid_points.iterrows():
        # 风速得分
        wind_speed = point.get('predicted_wind_speed', 0)
        wind_score = (wind_speed / max_wind_speed) * wind_speed_weight if max_wind_speed > 0 else 0

        # 利用率得分
        if 'wind_utilization_rate' in point:
            utilization_score = (point[
                                     'wind_utilization_rate'] / max_utilization) * utilization_weight if max_utilization > 0 else 0
        else:
            utilization_score = 0

        composite_score = wind_score + utilization_score
        composite_terms.append(composite_score * x[idx])

        # 成本项
        cost_value = 0
        if point.get('slope', 0) > kwargs.get('max_slope', 15):
            cost_value += point['slope'] * 10
        cost_terms.append(cost_value * x[idx])

    # 目标函数
    cost_weight = kwargs.get('cost_weight', 0.5)
    prob += pulp.lpSum(composite_terms) - cost_weight * pulp.lpSum(cost_terms)

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

    # 对于PuLP优化，储能优化需要单独处理（因为PuLP主要处理离散变量）
    storage_economic_analysis = {}
    if enable_storage_opt:
        # 可以在后续步骤中优化储能参数
        st.info("PuLP求解器主要用于离散优化，储能优化建议使用遗传算法或粒子群算法")

    return {
        'best_positions': selected_positions,
        'best_positions_data': best_positions_data,
        'best_fitness': best_fitness if best_fitness else 0,
        'fitness_history': [best_fitness] if best_fitness else [0],
        'algorithm': 'PuLP优化求解器（含储能优化）' if enable_storage_opt else 'PuLP优化求解器',
        'computation_time': computation_time,
        'power_results': power_results,
        'constraints_violated': check_constraints_violations(best_positions_data, kwargs),
        'optimization_weights': {
            'wind_speed_weight': wind_speed_weight,
            'utilization_weight': utilization_weight,
            'storage_weight': kwargs.get('storage_weight', 0.3) if enable_storage_opt else 0
        },
        'enable_storage_optimization': enable_storage_opt,
        'storage_economic_analysis': storage_economic_analysis
    }


def check_constraints_violations(positions_data, constraints):
    """检查约束违反情况"""
    violations = {}

    if positions_data.empty:
        return violations

    if 'slope' in positions_data.columns and 'max_slope' in constraints:
        slope_violations = positions_data[positions_data['slope'] > constraints['max_slope']]
        violations['slope'] = len(slope_violations)

    if 'road_distance' in positions_data.columns and 'max_road_distance' in constraints:
        road_violations = positions_data[positions_data['road_distance'] > constraints['max_road_distance']]
        violations['road'] = len(road_violations)

    # 添加其他约束检查...

    return violations


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
            result = real_genetic_algorithm_with_storage(df, **optimization_params)
        elif algo == "模拟退火算法":
            result = real_simulated_annealing(df, **optimization_params)
        elif algo == "粒子群优化算法":
            result = real_particle_swarm(df, **optimization_params)
        elif algo == "PuLP优化求解器":
            result = real_pulp_optimization(df, **optimization_params)
        else:
            result = real_genetic_algorithm_with_storage(df, **optimization_params)

        return result

    except Exception as e:
        st.error(f"优化算法执行错误: {str(e)}")
        # 回退到基础遗传算法
        st.info("尝试使用基础参数重新计算...")
        base_params = {
            'n_turbines': optimization_params.get('n_turbines', 5),
            'pop_size': 30,
            'generations': 50,
            'enable_storage_optimization': False
        }
        return real_genetic_algorithm(df, **base_params)


def call_optimize_function_with_all_strategies(df, algo, algorithm_params):
    """
    调用优化函数并测试所有储能策略 - 支持多风场
    """
    try:
        # 测试不同的储能策略
        strategies = ['平滑输出', '削峰填谷', '混合模式']
        strategy_results = []

        best_result = None
        best_fitness = -1
        best_strategy = None

        # 创建进度条和状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, strategy in enumerate(strategies):
            # 更新进度状态
            current_progress = (i + 1) / len(strategies)
            status_text.text(f"🔄 正在测试储能策略: {strategy} ({i + 1}/{len(strategies)})")
            progress_bar.progress(current_progress)

            # 更新策略参数
            current_params = algorithm_params.copy()
            current_params['storage_strategy'] = strategy

            # 调用优化函数
            result = call_optimize_function(df, algo, current_params)

            # 为每个风场生成独立的储能调度数据
            if 'best_positions' in result and len(result['best_positions']) > 0:
                # 获取风场数量
                n_farms = algorithm_params.get('n_farms', 1)
                n_turbines_per_farm = algorithm_params.get('n_turbines_per_farm',
                                                           len(result['best_positions']) // n_farms)

                # 分割风场数据
                farm_storage_results = []
                for farm_idx in range(n_farms):
                    # 计算当前风场的风机位置
                    start_idx = farm_idx * n_turbines_per_farm
                    end_idx = start_idx + n_turbines_per_farm
                    farm_positions = result['best_positions'][start_idx:end_idx]

                    # 为当前风场生成储能调度数据
                    storage_params = {
                        'storage_capacity': result.get('storage_economic_analysis', {}).get('storage_capacity_kwh',
                                                                                            60000),
                        'storage_power': result.get('storage_economic_analysis', {}).get('storage_power_kw', 30000),
                        'grid_capacity': current_params.get('grid_capacity', 20000),
                        'storage_strategy': strategy
                    }

                    farm_storage = generate_storage_schedule_data(df, farm_positions, storage_params)
                    farm_storage_results.append(farm_storage)

                # 存储所有风场的储能结果
                result['storage_results'] = farm_storage_results
                result['n_farms'] = n_farms

            # 记录策略结果
            fitness = result.get('best_fitness', 0)
            strategy_results.append({
                'strategy': strategy,
                'fitness': fitness,
                'computation_time': result.get('computation_time', 0),
                'quality_rating': evaluate_solution_quality(fitness)
            })

            # 更新最佳结果
            if fitness > best_fitness:
                best_fitness = fitness
                best_result = result
                best_strategy = strategy

        # 清理进度显示
        progress_bar.empty()
        status_text.empty()

        # 将策略比较结果添加到最佳结果中
        if best_result:
            best_result['strategy_comparison'] = strategy_results
            best_result['best_strategy'] = best_strategy
            best_result['best_fitness'] = best_fitness

            st.success(f"🏆 最佳储能策略: {best_strategy} (适应度: {best_fitness:.3f})")

        return best_result

    except Exception as e:
        st.error(f"多策略优化失败: {str(e)}")
        return call_optimize_function(df, algo, algorithm_params)


def evaluate_solution_quality(fitness):
    """
    简单评估解决方案质量
    """
    if fitness >= 900:
        return "🎯 优秀"
    elif fitness >= 800:
        return "🟢 良好"
    elif fitness >= 700:
        return "🟡 一般"
    else:
        return "🔴 需要改进"


def generate_storage_schedule_data(df, selected_positions, storage_params):
    """
    生成储能调度数据用于可视化
    """
    try:
        # 获取选中的风电场数据
        selected_data = df.loc[selected_positions]

        # 计算总风电功率时间序列
        time_series_data = calculate_wind_power_time_series(df, selected_data)

        # 应用储能调度策略
        storage_results = apply_storage_strategy(time_series_data, storage_params)

        return storage_results
    except Exception as e:
        st.error(f"生成储能调度数据时出错: {str(e)}")
        # 返回空的调度数据
        return {
            'schedule_data': pd.DataFrame(),
            'performance_metrics': {},
            'storage_params': storage_params,
            'strategy': storage_params.get('storage_strategy', '未知')
        }


def calculate_wind_power_time_series(df, selected_data):
    """
    基于原始数据计算风电功率时间序列
    """

    # 风速转功率函数
    def wind_speed_to_power(wind_speed):
        cut_in, rated, cut_out = 3.0, 12.5, 25.0
        rated_power = 2500  # kW
        if wind_speed < cut_in or wind_speed > cut_out:
            return 0
        elif wind_speed < rated:
            return rated_power * ((wind_speed - cut_in) / (rated - cut_in)) ** 3
        else:
            return rated_power

    # 获取选中的坐标点
    selected_points = selected_data['point_id'].unique()

    # 按时间聚合计算总功率
    time_series = []

    # 假设每个坐标点有4台风机
    turbines_per_point = 4

    # 按时间戳分组计算
    df_sorted = df.sort_values('timestamp')

    for timestamp in df_sorted['timestamp'].unique():
        # 获取该时间点所有选中坐标的数据
        time_data = df_sorted[
            (df_sorted['timestamp'] == timestamp) &
            (df_sorted['point_id'].isin(selected_points))
            ]

        if len(time_data) > 0:
            # 计算总风电功率
            total_power = 0
            for _, row in time_data.iterrows():
                power_per_turbine = wind_speed_to_power(row['predicted_wind_speed'])
                total_power += power_per_turbine * turbines_per_point

            # 提取时间信息
            hour = row['hour'] if 'hour' in row else 0
            minute = row['minute'] if 'minute' in row else 0

            time_series.append({
                'timestamp': timestamp,
                'hour': hour,
                'minute': minute,
                'time_index': len(time_series),
                'wind_power': total_power,
                'wind_speed_avg': time_data['predicted_wind_speed'].mean()
            })

    return pd.DataFrame(time_series)


def apply_storage_strategy(time_series_data, storage_params):
    """
    应用储能调度策略
    """
    storage_capacity = storage_params.get('storage_capacity', 60000)  # kWh
    max_power = storage_params.get('storage_power', 30000)  # kW
    grid_capacity = storage_params.get('grid_capacity', 20000)  # kW
    strategy = storage_params.get('storage_strategy', '平滑输出')

    # 初始化变量
    wind_power = time_series_data['wind_power'].values
    n_periods = len(wind_power)

    battery_power = np.zeros(n_periods)  # 正值放电，负值充电
    soc = np.zeros(n_periods)  # 荷电状态 (0-1)
    grid_power = np.zeros(n_periods)  # 并网功率
    wind_curtailment = np.zeros(n_periods)  # 弃风功率

    # 初始SOC
    soc[0] = 0.5  # 50%初始电量

    # 根据策略选择参数
    if strategy == '平滑输出':
        smoothing_factor = 0.8
        peak_threshold = 0.9
    elif strategy == '削峰填谷':
        smoothing_factor = 0.6
        peak_threshold = 0.8
    else:  # 混合模式
        smoothing_factor = 0.7
        peak_threshold = 0.85

    for t in range(n_periods):
        current_wind_power = wind_power[t]

        # 计算功率差额
        power_diff = current_wind_power - grid_capacity

        if strategy == '平滑输出':
            # 平滑输出策略
            if power_diff > 0:  # 风电过剩
                # 充电
                charge_power = min(power_diff, max_power,
                                   storage_capacity * (0.9 - soc[t - 1]) / 0.95 if t > 0 else max_power)
                battery_power[t] = -charge_power
                grid_power[t] = grid_capacity
                wind_curtailment[t] = power_diff - charge_power
            else:  # 风电不足
                # 放电
                discharge_power = min(-power_diff, max_power,
                                      (soc[t - 1] - 0.1) * storage_capacity * 0.95 if t > 0 else max_power)
                battery_power[t] = discharge_power
                grid_power[t] = current_wind_power + discharge_power
                wind_curtailment[t] = 0

        elif strategy == '削峰填谷':
            # 削峰填谷策略
            if current_wind_power > grid_capacity * peak_threshold:  # 高峰时段
                charge_power = min(current_wind_power - grid_capacity * peak_threshold, max_power,
                                   storage_capacity * (0.9 - soc[t - 1]) / 0.95 if t > 0 else max_power)
                battery_power[t] = -charge_power
                grid_power[t] = grid_capacity * peak_threshold
                wind_curtailment[t] = current_wind_power - grid_capacity * peak_threshold - charge_power
            elif current_wind_power < grid_capacity * 0.6:  # 低谷时段
                discharge_power = min(grid_capacity * 0.6 - current_wind_power, max_power,
                                      (soc[t - 1] - 0.1) * storage_capacity * 0.95 if t > 0 else max_power)
                battery_power[t] = discharge_power
                grid_power[t] = current_wind_power + discharge_power
                wind_curtailment[t] = 0
            else:  # 正常时段
                battery_power[t] = 0
                grid_power[t] = current_wind_power
                wind_curtailment[t] = 0

        else:  # 混合模式
            # 混合模式：结合平滑输出和削峰填谷的优点
            if power_diff > 0:  # 风电过剩
                # 根据SOC决定充电策略
                if t > 0 and soc[t - 1] < 0.7:  # SOC较低时多充电
                    charge_power = min(power_diff * smoothing_factor, max_power,
                                       storage_capacity * (0.9 - soc[t - 1]) / 0.95)
                else:  # SOC较高时少充电
                    charge_power = min(power_diff * 0.5, max_power,
                                       storage_capacity * (0.9 - soc[t - 1]) / 0.95)

                battery_power[t] = -charge_power
                grid_power[t] = current_wind_power - charge_power
                wind_curtailment[t] = max(0, power_diff - charge_power)

            else:  # 风电不足
                # 根据SOC决定放电策略
                if t > 0 and soc[t - 1] > 0.4:  # SOC较高时多放电
                    discharge_power = min(-power_diff, max_power,
                                          (soc[t - 1] - 0.1) * storage_capacity * 0.95)
                else:  # SOC较低时少放电
                    discharge_power = min(-power_diff * 0.5, max_power,
                                          (soc[t - 1] - 0.1) * storage_capacity * 0.95)

                battery_power[t] = discharge_power
                grid_power[t] = current_wind_power + discharge_power
                wind_curtailment[t] = 0

        # 更新SOC
        if t > 0:
            energy_change = -battery_power[t] * (10 / 60)  # 10分钟间隔，转换为kWh
            soc[t] = max(0.1, min(0.9, soc[t - 1] + energy_change / storage_capacity))
        else:
            energy_change = -battery_power[t] * (10 / 60)
            soc[t] = max(0.1, min(0.9, 0.5 + energy_change / storage_capacity))

    # 创建结果数据框
    result_df = time_series_data.copy()
    result_df['battery_power'] = battery_power
    result_df['grid_power'] = grid_power
    result_df['storage_soc'] = soc
    result_df['wind_curtailment'] = wind_curtailment
    result_df['net_power'] = result_df['wind_power'] + result_df['battery_power']

    # 计算性能指标
    performance_metrics = calculate_storage_performance(result_df, storage_params)

    return {
        'schedule_data': result_df,
        'performance_metrics': performance_metrics,
        'storage_params': storage_params,
        'strategy': strategy
    }


def calculate_storage_performance(storage_data, storage_params):
    """
    计算储能系统性能指标
    """
    wind_power = storage_data['wind_power']
    grid_power = storage_data['grid_power']
    battery_power = storage_data['battery_power']
    storage_capacity = storage_params.get('storage_capacity', 60000)

    # 平滑效果
    wind_fluctuation = wind_power.std()
    grid_fluctuation = grid_power.std()
    smoothing_effect = ((wind_fluctuation - grid_fluctuation) / wind_fluctuation * 100) if wind_fluctuation > 0 else 0

    # 储能利用率
    total_charge = abs(storage_data[storage_data['battery_power'] < 0]['battery_power'].sum() * (10 / 60))
    total_discharge = storage_data[storage_data['battery_power'] > 0]['battery_power'].sum() * (10 / 60)
    storage_utilization = (total_charge + total_discharge) / (2 * storage_capacity) * 100

    # 弃风率
    total_wind_energy = wind_power.sum() * (10 / 60)
    total_curtailment = storage_data['wind_curtailment'].sum() * (10 / 60)
    curtailment_rate = (total_curtailment / total_wind_energy * 100) if total_wind_energy > 0 else 0

    # 系统效率（假设充放电效率为95%）
    system_efficiency = (total_discharge / total_charge * 100) if total_charge > 0 else 0

    return {
        'smoothing_effect': smoothing_effect,
        'storage_utilization': storage_utilization,
        'curtailment_rate': curtailment_rate,
        'system_efficiency': system_efficiency,
        'total_charge_energy': total_charge,
        'total_discharge_energy': total_discharge,
        'wind_fluctuation': wind_fluctuation,
        'grid_fluctuation': grid_fluctuation
    }


# ==============================
# 新的储能优化专用函数
# ==============================

def optimize_storage_only(df, wind_farm_positions, **kwargs):
    """
    单独优化储能参数（给定风机位置）

    参数：
    - df: 完整数据集
    - wind_farm_positions: 已确定的风机位置
    - kwargs: 储能优化参数

    返回：
    - 优化的储能参数和经济性分析
    """
    start_time = time.time()

    # 获取风机位置数据
    selected_data = df.loc[wind_farm_positions]

    # 储能参数范围
    min_capacity = kwargs.get('min_storage_capacity', 10000)  # kWh
    max_capacity = kwargs.get('max_storage_capacity', 200000)  # kWh
    min_power = kwargs.get('min_storage_power', 5000)  # kW
    max_power = kwargs.get('max_storage_power', 100000)  # kW

    # 使用简单搜索方法优化储能参数
    n_samples = kwargs.get('storage_samples', 20)
    best_storage_params = None
    best_economic_score = -float('inf')

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(n_samples):
        # 生成随机的储能参数
        capacity = np.random.uniform(min_capacity, max_capacity)
        power = np.random.uniform(min_power, max_power)

        # 计算储能经济性
        investment_cost = calculate_storage_investment_cost(capacity, power)
        annual_revenue = calculate_storage_annual_revenue(selected_data, capacity, power, kwargs)
        om_cost = calculate_storage_operation_cost(capacity, power)
        net_annual_benefit = annual_revenue - om_cost

        # 经济性评分（净现值简化版）
        discount_rate = 0.08  # 8%折现率
        project_life = 20  # 项目寿命20年

        # 计算净现值
        npv = -investment_cost
        for year in range(1, project_life + 1):
            npv += net_annual_benefit / ((1 + discount_rate) ** year)

        # 更新最优结果
        if npv > best_economic_score:
            best_economic_score = npv
            best_storage_params = {
                'capacity_kwh': capacity,
                'power_kw': power,
                'investment_cost': investment_cost,
                'annual_revenue': annual_revenue,
                'om_cost': om_cost,
                'net_annual_benefit': net_annual_benefit,
                'npv': npv,
                'payback_years': investment_cost / net_annual_benefit if net_annual_benefit > 0 else float('inf')
            }

        # 更新进度
        progress = (i + 1) / n_samples
        progress_bar.progress(progress)
        status_text.text(f"储能优化进度: {i + 1}/{n_samples}, 当前最佳NPV: {best_economic_score:.2f} 万元")

    progress_bar.empty()
    status_text.empty()

    computation_time = time.time() - start_time

    return {
        'optimized_storage_params': best_storage_params,
        'computation_time': computation_time,
        'wind_farm_data': selected_data,
        'n_samples': n_samples
    }


def two_level_optimization(df, n_turbines, **kwargs):
    """
    两层优化框架：
    1. 外层：优化风机位置（不含储能）
    2. 内层：给定风机位置，优化储能参数

    返回：
    - 最优风机位置
    - 最优储能参数
    - 综合经济性分析
    """
    st.info("🚀 开始两层优化：先优化风机位置，再优化储能配置")

    # 第一层：优化风机位置（不含储能）
    st.subheader("第一层：风机位置优化")
    wind_farm_params = kwargs.copy()
    wind_farm_params['enable_storage_optimization'] = False

    # 调用遗传算法优化风机位置
    wind_farm_result = real_genetic_algorithm(df, n_turbines, **wind_farm_params)

    if wind_farm_result is None or len(wind_farm_result.get('best_positions', [])) == 0:
        st.error("风机位置优化失败")
        return None

    st.success(f"✅ 风机位置优化完成，最优适应度: {wind_farm_result['best_fitness']:.2f}")

    # 第二层：优化储能参数
    st.subheader("第二层：储能配置优化")
    best_wind_positions = wind_farm_result['best_positions']

    # 优化储能参数
    storage_opt_result = optimize_storage_only(df, best_wind_positions, **kwargs)

    if storage_opt_result is None or storage_opt_result.get('optimized_storage_params') is None:
        st.error("储能参数优化失败")
        return wind_farm_result

    st.success(f"✅ 储能配置优化完成，最优NPV: {storage_opt_result['optimized_storage_params']['npv']:.2f} 万元")

    # 合并结果
    combined_result = wind_farm_result.copy()
    combined_result['storage_optimization_result'] = storage_opt_result
    combined_result['algorithm'] = '两层优化算法'
    combined_result['optimization_type'] = '风机位置 + 储能配置'

    # 计算综合经济性
    wind_economic = wind_farm_result.get('power_results', {}).get('economic_analysis', {})
    storage_economic = storage_opt_result.get('optimized_storage_params', {})

    total_investment = wind_economic.get('total_investment', 0) + storage_economic.get('investment_cost', 0)
    total_annual_revenue = wind_economic.get('annual_revenue', 0) + storage_economic.get('annual_revenue', 0)
    total_annual_profit = wind_economic.get('annual_profit', 0) + storage_economic.get('net_annual_benefit', 0)

    combined_result['combined_economic_analysis'] = {
        'total_investment': total_investment,
        'total_annual_revenue': total_annual_revenue,
        'total_annual_profit': total_annual_profit,
        'combined_payback_years': total_investment / total_annual_profit if total_annual_profit > 0 else float('inf'),
        'combined_npv': wind_economic.get('annual_profit', 0) * 10 + storage_economic.get('npv', 0)  # 简化计算
    }

    return combined_result